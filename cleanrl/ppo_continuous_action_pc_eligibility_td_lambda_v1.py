# Predictive-coding actor-critic with streaming TD(lambda) eligibility, v1.
#
# Each vector transition updates immediately: no rollout buffer, GAE, PPO, replay,
# or backpropagation through the hidden hierarchy. Actor and critic hidden states
# settle under the standard Gaussian PC energy, W f(z), using a unit terminal
# direction independent of the current TD error. Layer-scalar curvature steps keep
# momentum inference stable without destroying the magnitude of a weak terminal
# signal. Exact head scores and local
# precision-weighted residual scores enter separate per-environment accumulating
# eligibility traces; the subsequent TD error modulates those traces before done
# rows are reset. This is standard PC plus temporal eligibility, not Bayesian PC.
# The state equation and precision-error signs follow Appendix C of arXiv:2503.24016;
# its Matrix-Normal-Wishart BPC posterior is deliberately reserved for a future arm.
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions import kl_divergence
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter


SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 16
    gamma: float = 0.99
    trace_lambda: float = 0.95
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    local_learning_rate: float = 3e-4
    anneal_lr: bool = True
    vf_coef: float = 0.5
    td_rms_decay: float = 0.999
    td_rms_min: float = 0.1
    actor_td_clip: float = 10.0
    critic_td_clip: float = 10.0
    max_direction_norm: float = 0.5
    log_interval: int = 100

    hidden_size: int = 64
    pc_num_hidden_layers: int = 6
    pc_inference_steps: int = 50
    pc_inference_scale: float = 0.1
    """dimensionless numerator for each layer step, scale / local curvature bound"""
    pc_momentum: float = 0.65
    pc_convergence_tol: float = 0.0
    """optional early-stop tolerance; zero runs fixed max steps without device synchronization"""
    pc_activation_grad_clip: float = 5.0
    pc_state_clip: float = 5.0
    pc_actor_terminal_coef: float = 1.0
    pc_critic_terminal_coef: float = 1.0
    pc_input_activation: str = "identity"
    pc_hidden_activation: str = "tanh"
    pc_initial_precision: float = 1.0
    pc_precision_ema: float = 0.001
    pc_precision_ridge: float = 0.01
    pc_precision_min: float = 0.1
    pc_precision_max: float = 10.0
    pc_curvature_safety: float = 1.05
    pc_curvature_refresh_interval: int = 1000

    target_kl: float = 0.003
    max_kl: float = 0.01
    kl_adapt_rate: float = 0.05
    kl_scale_min: float = 0.05
    kl_scale_max: float = 2.0

    compile: bool = False
    compile_mode: Optional[str] = "reduce-overhead"
    pc_compile_chunk_steps: int = 5
    """PC sweeps fused into each compiled inference chunk"""
    torch_float32_matmul_precision: str = "high"
    num_updates: int = 0


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


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def activation(x, name):
    if name == "identity":
        return x
    if name == "tanh":
        return torch.tanh(x)
    if name == "silu":
        return F.silu(x)
    raise ValueError(f"unknown activation {name!r}")


def activation_derivative(x, name):
    if name == "identity":
        return torch.ones_like(x)
    if name == "tanh":
        y = torch.tanh(x)
        return 1.0 - y.square()
    if name == "silu":
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1.0 + x * (1.0 - sigmoid))
    raise ValueError(f"unknown activation {name!r}")


def make_compiled_settle_core(args, activation_names, terminal_kind):
    """Build one bounded fixed-shape graph for a chunk of PC sweeps."""
    num_layers = len(activation_names)
    num_steps = args.pc_compile_chunk_steps
    momentum = args.pc_momentum
    grad_clip = args.pc_activation_grad_clip
    state_clip = args.pc_state_clip
    actor_coef = args.pc_actor_terminal_coef
    critic_coef = args.pc_critic_terminal_coef

    def core(x, input_states, input_moments, weights, biases, precisions, layer_steps, terminal_inputs):
        states = list(input_states)
        moments = list(input_moments)
        for _ in range(num_steps):
            for layer_idx in reversed(range(num_layers)):
                source = x if layer_idx == 0 else states[layer_idx - 1]
                mean = F.linear(activation(source, activation_names[layer_idx]), weights[layer_idx], biases[layer_idx])
                error = states[layer_idx] - mean
                gradient = error * precisions[layer_idx]
                if layer_idx + 1 < num_layers:
                    downstream_mean = F.linear(
                        activation(states[layer_idx], activation_names[layer_idx + 1]),
                        weights[layer_idx + 1],
                        biases[layer_idx + 1],
                    )
                    downstream_error = (states[layer_idx + 1] - downstream_mean) * precisions[layer_idx + 1]
                    correction = F.linear(downstream_error, weights[layer_idx + 1].T)
                    gradient = gradient - activation_derivative(
                        states[layer_idx], activation_names[layer_idx + 1]
                    ) * correction
                elif terminal_kind == "actor":
                    action_z, alpha_weight, alpha_bias, beta_weight, beta_bias = terminal_inputs
                    alpha_raw = F.linear(states[layer_idx], alpha_weight, alpha_bias)
                    beta_raw = F.linear(states[layer_idx], beta_weight, beta_bias)
                    alpha = 1.0 + F.softplus(alpha_raw)
                    beta = 1.0 + F.softplus(beta_raw)
                    z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    common = torch.digamma(alpha + beta)
                    alpha_score = (z.log() - torch.digamma(alpha) + common) * torch.sigmoid(alpha_raw)
                    beta_score = ((1.0 - z).log() - torch.digamma(beta) + common) * torch.sigmoid(beta_raw)
                    logprob_gradient = F.linear(alpha_score, alpha_weight.T)
                    logprob_gradient = logprob_gradient + F.linear(beta_score, beta_weight.T)
                    gradient = gradient - actor_coef * logprob_gradient
                else:
                    (critic_weight,) = terminal_inputs
                    gradient = gradient - critic_coef * critic_weight.expand(states[layer_idx].shape[0], -1)
                if grad_clip > 0:
                    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    gradient = gradient * (grad_clip / norm).clamp(max=1.0)
                moments[layer_idx] = momentum * moments[layer_idx] + gradient
                states[layer_idx] = (
                    states[layer_idx] - layer_steps[layer_idx] * moments[layer_idx]
                ).clamp(-state_clip, state_clip)
        return tuple(states), tuple(moments)

    import torch._inductor as inductor

    compile_options = dict(inductor.list_mode_options(args.compile_mode, dynamic=False))
    # State/momentum outputs feed the next chunk and final states remain live while
    # the other pathway runs. Inductor fusion supplies the speedup; CUDA graph
    # output reuse is unsafe for these long-lived mutable-state tensors.
    compile_options["triton.cudagraphs"] = False
    return torch.compile(core, dynamic=False, fullgraph=True, options=compile_options)


def bootstrap_observations(next_obs, truncations, infos):
    """Use final observations for time limits; autoreset observations are wrong here."""
    bootstrap_obs = np.array(next_obs, copy=True)
    if not np.any(truncations):
        return bootstrap_obs
    finals = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if finals is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")
    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        if finals[env_idx] is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = finals[env_idx]
    return bootstrap_obs


class RunningTDRMS:
    def __init__(self, device, decay, minimum):
        self.decay = decay
        self.minimum = minimum
        self.mean_square = torch.ones((), device=device)
        self.initialized = False

    @torch.no_grad()
    def normalize(self, delta, clip):
        square = delta.square().mean()
        if self.initialized:
            self.mean_square.lerp_(square, 1.0 - self.decay)
        else:
            self.mean_square.copy_(square)
            self.initialized = True
        normalized = delta / self.mean_square.sqrt().clamp_min(self.minimum)
        return normalized.clamp(-clip, clip) if clip > 0 else normalized


class LocalPredictor(nn.Module):
    """Gaussian edge z_l ~ N(W_l f(z_{l-1}) + b_l, diag(P_l)^-1)."""

    def __init__(self, in_dim, out_dim, activation_name, args):
        super().__init__()
        linear = layer_init(nn.Linear(in_dim, out_dim))
        self.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
        self.activation_name = activation_name
        self.register_buffer("precision", torch.full((out_dim,), args.pc_initial_precision))
        self.register_buffer("residual_second_moment", torch.full((out_dim,), 1.0 / args.pc_initial_precision))

    def features(self, source):
        return activation(source, self.activation_name)

    def augmented_features(self, source):
        features = self.features(source)
        return torch.cat([features, torch.ones_like(features[:, :1])], dim=1)

    def forward(self, source):
        return F.linear(self.features(source), self.weight, self.bias)

    def residual(self, source, target):
        return target - self(source)

    def precision_residual(self, source, target):
        return self.residual(source, target) * self.precision

    @torch.no_grad()
    def update_precision(self, residual, args):
        if args.pc_precision_ema <= 0:
            return
        batch_second = residual.detach().square().mean(dim=0)
        self.residual_second_moment.lerp_(batch_second, args.pc_precision_ema)
        new_precision = (self.residual_second_moment + args.pc_precision_ridge).reciprocal()
        self.precision.copy_(new_precision.clamp(args.pc_precision_min, args.pc_precision_max))


class PCHierarchy(nn.Module):
    def __init__(self, input_dim, hidden_size, args):
        super().__init__()
        assert args.pc_num_hidden_layers >= 1
        edges = [LocalPredictor(input_dim, hidden_size, args.pc_input_activation, args)]
        edges.extend(
            LocalPredictor(hidden_size, hidden_size, args.pc_hidden_activation, args)
            for _ in range(1, args.pc_num_hidden_layers)
        )
        self.edges = nn.ModuleList(edges)
        self.register_buffer("cached_curvature_bounds", torch.ones(len(edges)))
        self._curvature_initialized = False
        self._curvature_cache_age = args.pc_curvature_refresh_interval

    @torch.no_grad()
    def initial_states(self, x, args):
        states = []
        source = x
        for edge in self.edges:
            state = edge(source).clamp(-args.pc_state_clip, args.pc_state_clip)
            states.append(state)
            source = state
        return states

    def residuals(self, x, states):
        residuals = []
        source = x
        for edge, state in zip(self.edges, states):
            residuals.append(edge.residual(source, state))
            source = state
        return residuals

    def activation_names(self):
        return tuple(edge.activation_name for edge in self.edges)

    def compiled_settle(self, x, compiled_core, terminal_inputs, args, collect_diagnostics):
        initial_states = tuple(self.initial_states(x, args))
        curvature_bounds = self.curvature_bounds(args)
        layer_steps = args.pc_inference_scale / curvature_bounds
        states = initial_states
        moments = tuple(torch.zeros_like(state) for state in states)
        weights = tuple(edge.weight for edge in self.edges)
        biases = tuple(edge.bias for edge in self.edges)
        precisions = tuple(edge.precision for edge in self.edges)
        if args.pc_inference_steps % args.pc_compile_chunk_steps != 0:
            raise ValueError("pc_inference_steps must be divisible by pc_compile_chunk_steps")
        for _ in range(args.pc_inference_steps // args.pc_compile_chunk_steps):
            states, moments = compiled_core(
                x,
                states,
                moments,
                weights,
                biases,
                precisions,
                layer_steps,
                terminal_inputs,
            )
        residuals = tuple(self.residuals(x, states))
        diagnostics = {
            "steps": args.pc_inference_steps,
            "mean_step": layer_steps.mean(),
            "max_curvature": curvature_bounds.max(),
            "displacements": (
                [(state - base).norm(dim=1) for state, base in zip(states, initial_states)]
                if collect_diagnostics else []
            ),
            "residuals": (
                [residual.norm(dim=1) for residual in residuals] if collect_diagnostics else []
            ),
        }
        return list(states), list(residuals), diagnostics

    @torch.no_grad()
    def curvature_bounds(self, args):
        """Cache layer-scalar curvature estimates for many streaming updates.

        A sparse exact symmetric eigensolve obtains the downstream spectral term;
        adding max(P_l) yields a batch-safe layer-curvature upper bound. A small safety multiplier and
        slow clipped 3e-4 parameter motion make the cached value conservative
        between refreshes, avoiding both per-update eigensolves and the unusably
        loose induced-norm bound.
        """
        if self._curvature_initialized:
            self._curvature_cache_age += 1
            if self._curvature_cache_age < args.pc_curvature_refresh_interval:
                return self.cached_curvature_bounds
        bounds = []
        for layer_idx, edge in enumerate(self.edges):
            if layer_idx + 1 == len(self.edges):
                bounds.append((args.pc_curvature_safety * edge.precision.max()).clamp_min(1e-8))
                continue
            downstream = self.edges[layer_idx + 1]
            downstream_curvature = downstream.weight.T @ (
                downstream.precision.unsqueeze(1) * downstream.weight
            )
            # f'(z)^2 <= 1 for identity/tanh, so p_l,max plus this
            # downstream spectral term is a valid batch-independent upper bound.
            largest_eigenvalue = edge.precision.max() + torch.linalg.eigvalsh(
                downstream_curvature
            ).max()
            bounds.append((args.pc_curvature_safety * largest_eigenvalue).clamp_min(1e-8))
        self.cached_curvature_bounds.copy_(torch.stack(bounds))
        self._curvature_initialized = True
        self._curvature_cache_age = 0
        return self.cached_curvature_bounds

    def settle(self, x, terminal_grad_fn, args, collect_diagnostics=True):
        """Reverse Gauss-Seidel momentum inference with scalar curvature steps.

        For layer l, a sparsely refreshed eigensolve estimates the largest
        eigenvalue of the local precision plus downstream curvature. The cached
        layer-scalar step preserves terminal-error proportionality, unlike
        coordinatewise normalization, while repeated sweeps avoid six-layer
        finite-step extinction.
        """
        states = [state.clone() for state in self.initial_states(x, args)]
        deployed = [state.clone() for state in states]
        first_moments = [torch.zeros_like(state) for state in states]
        curvature_bounds = self.curvature_bounds(args)
        layer_steps = [args.pc_inference_scale / bound for bound in curvature_bounds]
        steps_used = 0
        for step in range(args.pc_inference_steps):
            max_move = states[0].new_zeros(()) if args.pc_convergence_tol > 0 else None
            # Top-down order makes a terminal perturbation reach every layer in one
            # sweep, while each correction uses the newest downstream state.
            for layer_idx in reversed(range(len(states))):
                source = x if layer_idx == 0 else states[layer_idx - 1]
                edge = self.edges[layer_idx]
                eps = edge.residual(source, states[layer_idx])
                grad = eps * edge.precision
                if layer_idx + 1 < len(states):
                    downstream = self.edges[layer_idx + 1]
                    down_eps = downstream.precision_residual(states[layer_idx], states[layer_idx + 1])
                    correction = F.linear(down_eps, downstream.weight.T)
                    grad = grad - activation_derivative(states[layer_idx], downstream.activation_name) * correction
                else:
                    grad = grad + terminal_grad_fn(states[layer_idx])
                if args.pc_activation_grad_clip > 0:
                    norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    grad = grad * (args.pc_activation_grad_clip / norm).clamp(max=1.0)
                first_moments[layer_idx] = args.pc_momentum * first_moments[layer_idx] + grad
                old = states[layer_idx]
                states[layer_idx] = (old - layer_steps[layer_idx] * first_moments[layer_idx]).clamp(
                    -args.pc_state_clip, args.pc_state_clip
                )
                if max_move is not None:
                    max_move = torch.maximum(max_move, (states[layer_idx] - old).abs().max())
            steps_used = step + 1
            if max_move is not None and float(max_move) < args.pc_convergence_tol:
                break
        residuals = self.residuals(x, states)
        diagnostics = {
            "steps": steps_used,
            "mean_step": torch.stack(layer_steps).mean(),
            "max_curvature": curvature_bounds.max(),
            "displacements": (
                [(state - base).norm(dim=1) for state, base in zip(states, deployed)]
                if collect_diagnostics else []
            ),
            "residuals": [residual.norm(dim=1) for residual in residuals] if collect_diagnostics else [],
        }
        return states, residuals, diagnostics

    @torch.no_grad()
    def local_scores(self, x, states):
        scores = []
        source = x
        for edge, state in zip(self.edges, states):
            phi = edge.augmented_features(source)
            precision_residual = edge.precision_residual(source, state)
            scores.append(precision_residual.unsqueeze(2) * phi.unsqueeze(1))
            source = state
        return scores

    @torch.no_grad()
    def update_precisions(self, residuals, args):
        """Update from residuals cached before any parameter mutation."""
        assert len(residuals) == len(self.edges)
        for edge, residual in zip(self.edges, residuals):
            edge.update_precision(residual, args)


def beta_head_scores(alpha_raw, beta_raw, alpha, beta, action_z):
    """Exact per-example score with respect to the two unconstrained head outputs."""
    z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    common = torch.digamma(alpha + beta)
    alpha_score = (z.log() - torch.digamma(alpha) + common) * torch.sigmoid(alpha_raw)
    beta_score = ((1.0 - z).log() - torch.digamma(beta) + common) * torch.sigmoid(beta_raw)
    return alpha_score, beta_score


def linear_scores(output_score, features):
    return output_score.unsqueeze(2) * features.unsqueeze(1), output_score


class TraceBank:
    """A collection of independent per-environment accumulating traces."""

    def __init__(self, templates):
        self.traces = [torch.zeros_like(template) for template in templates]

    @torch.no_grad()
    def accumulate(self, scores, decay):
        assert len(scores) == len(self.traces)
        for trace, score in zip(self.traces, scores):
            trace.mul_(decay).add_(score)

    @torch.no_grad()
    def modulated_mean(self, delta):
        # Delta is associated per environment before the only environment reduction.
        return [torch.einsum("b,b...->...", delta, trace) / delta.shape[0] for trace in self.traces]

    @torch.no_grad()
    def reset(self, done):
        for trace in self.traces:
            trace[done] = 0

    def layer_rms(self):
        return [trace.square().mean().sqrt() for trace in self.traces]


def clip_directions(directions, max_norm):
    if max_norm <= 0:
        return directions, torch.zeros(())
    norm = torch.stack([direction.float().square().sum() for direction in directions]).sum().sqrt()
    scale = (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    return [direction * scale for direction in directions], norm


def direction_norm(directions):
    return torch.stack([direction.float().square().sum() for direction in directions]).sum().sqrt()


class KLController:
    def __init__(self, args):
        self.scale = 1.0
        self.target = args.target_kl
        self.maximum = args.max_kl
        self.adapt_rate = args.kl_adapt_rate
        self.minimum_scale = args.kl_scale_min
        self.maximum_scale = args.kl_scale_max

    def hard_scale(self, measured_kl):
        if measured_kl <= self.maximum:
            return 1.0
        return math.sqrt(self.maximum / max(measured_kl, 1e-12))

    def adapt(self, measured_kl):
        signed_error = (self.target - measured_kl) / max(self.target, 1e-12)
        self.scale *= math.exp(self.adapt_rate * max(-2.0, min(2.0, signed_error)))
        self.scale = max(self.minimum_scale, min(self.maximum_scale, self.scale))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.actor_alpha_head = layer_init(nn.Linear(args.hidden_size, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(args.hidden_size, act_dim), std=0.01)
        self.critic_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.critic_head = layer_init(nn.Linear(args.hidden_size, 1), std=1.0)
        self._compiled_actor_settle = None
        self._compiled_critic_settle = None
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def compile_inference(self, args):
        if not args.compile:
            return
        if args.pc_convergence_tol > 0:
            raise ValueError("compiled fixed-step inference requires pc_convergence_tol=0")
        if args.pc_compile_chunk_steps <= 0 or args.pc_inference_steps % args.pc_compile_chunk_steps != 0:
            raise ValueError("pc_inference_steps must be divisible by positive pc_compile_chunk_steps")
        self._compiled_actor_settle = make_compiled_settle_core(
            args, self.actor_pc.activation_names(), "actor"
        )
        self._compiled_critic_settle = make_compiled_settle_core(
            args, self.critic_pc.activation_names(), "critic"
        )

    def actor_outputs(self, h):
        alpha_raw = self.actor_alpha_head(h)
        beta_raw = self.actor_beta_head(h)
        alpha = 1.0 + F.softplus(alpha_raw)
        beta = 1.0 + F.softplus(beta_raw)
        return alpha_raw, beta_raw, Beta(alpha, beta)

    def deployment(self, hierarchy, x, args):
        return hierarchy.initial_states(x, args)

    def get_value(self, x, args):
        return self.critic_head(self.deployment(self.critic_pc, x, args)[-1]).view(-1)

    def act(self, x, args):
        actor_states = self.deployment(self.actor_pc, x, args)
        critic_states = self.deployment(self.critic_pc, x, args)
        alpha_raw, beta_raw, dist = self.actor_outputs(actor_states[-1])
        z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        value = self.critic_head(critic_states[-1]).view(-1)
        return action, z, value, actor_states, critic_states, alpha_raw, beta_raw, dist

    def actor_terminal_grad(self, h, action_z, args):
        alpha_raw, beta_raw, dist = self.actor_outputs(h)
        alpha_score, beta_score = beta_head_scores(
            alpha_raw, beta_raw, dist.concentration1, dist.concentration0, action_z
        )
        logprob_grad = F.linear(alpha_score, self.actor_alpha_head.weight.T)
        logprob_grad = logprob_grad + F.linear(beta_score, self.actor_beta_head.weight.T)
        return -args.pc_actor_terminal_coef * logprob_grad

    def critic_terminal_grad(self, h, args):
        return -args.pc_critic_terminal_coef * self.critic_head.weight.expand(h.shape[0], -1)

    def settle_actor(self, x, action_z, args, collect_diagnostics=True):
        if self._compiled_actor_settle is not None:
            return self.actor_pc.compiled_settle(
                x,
                self._compiled_actor_settle,
                (
                    action_z,
                    self.actor_alpha_head.weight,
                    self.actor_alpha_head.bias,
                    self.actor_beta_head.weight,
                    self.actor_beta_head.bias,
                ),
                args,
                collect_diagnostics,
            )
        return self.actor_pc.settle(
            x, lambda h: self.actor_terminal_grad(h, action_z, args), args, collect_diagnostics
        )

    def settle_critic(self, x, args, collect_diagnostics=True):
        if self._compiled_critic_settle is not None:
            return self.critic_pc.compiled_settle(
                x,
                self._compiled_critic_settle,
                (self.critic_head.weight,),
                args,
                collect_diagnostics,
            )
        return self.critic_pc.settle(
            x, lambda h: self.critic_terminal_grad(h, args), args, collect_diagnostics
        )

    @torch.no_grad()
    def snapshot_actor(self):
        tensors = [self.actor_alpha_head.weight, self.actor_alpha_head.bias,
                   self.actor_beta_head.weight, self.actor_beta_head.bias]
        for edge in self.actor_pc.edges:
            tensors.extend([edge.weight, edge.bias])
        return [tensor.clone() for tensor in tensors]

    @torch.no_grad()
    def restore_actor(self, snapshot):
        tensors = [self.actor_alpha_head.weight, self.actor_alpha_head.bias,
                   self.actor_beta_head.weight, self.actor_beta_head.bias]
        for edge in self.actor_pc.edges:
            tensors.extend([edge.weight, edge.bias])
        for tensor, old in zip(tensors, snapshot):
            tensor.copy_(old)


@torch.no_grad()
def apply_head_directions(agent, directions, lr, actor):
    if actor:
        tensors = [agent.actor_alpha_head.weight, agent.actor_alpha_head.bias,
                   agent.actor_beta_head.weight, agent.actor_beta_head.bias]
    else:
        tensors = [agent.critic_head.weight, agent.critic_head.bias]
    for tensor, direction in zip(tensors, directions):
        tensor.add_(direction, alpha=lr)


@torch.no_grad()
def apply_local_directions(hierarchy, directions, lr):
    for edge, direction in zip(hierarchy.edges, directions):
        edge.weight.add_(direction[:, :-1], alpha=lr)
        edge.bias.add_(direction[:, -1], alpha=lr)


def make_trace_banks(agent, args, device):
    batch = args.num_envs
    act_dim = agent.actor_alpha_head.out_features
    hidden = args.hidden_size
    actor_head = TraceBank([
        torch.empty(batch, act_dim, hidden, device=device), torch.empty(batch, act_dim, device=device),
        torch.empty(batch, act_dim, hidden, device=device), torch.empty(batch, act_dim, device=device),
    ])
    critic_head = TraceBank([
        torch.empty(batch, 1, hidden, device=device), torch.empty(batch, 1, device=device),
    ])
    actor_local = TraceBank([
        torch.empty(batch, edge.weight.shape[0], edge.weight.shape[1] + 1, device=device)
        for edge in agent.actor_pc.edges
    ])
    critic_local = TraceBank([
        torch.empty(batch, edge.weight.shape[0], edge.weight.shape[1] + 1, device=device)
        for edge in agent.critic_pc.edges
    ])
    return actor_head, critic_head, actor_local, critic_local


def main():
    args = tyro.cli(Args)
    args.num_updates = args.total_timesteps // args.num_envs
    assert args.cuda and torch.cuda.is_available(), "CUDA is required for this research variant"
    if args.compile:
        print("compile requested; fixed-shape PC inference will be compiled, trace bookkeeping remains eager")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(
        f"|{key}|{value}|" for key, value in vars(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)
    device = torch.device("cuda")
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, idx, args.capture_video, run_name, args.gamma) for idx in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    agent = Agent(envs, args).to(device)
    agent.compile_inference(args)
    actor_head_trace, critic_head_trace, actor_local_trace, critic_local_trace = make_trace_banks(
        agent, args, device)
    traces = (actor_head_trace, critic_head_trace, actor_local_trace, critic_local_trace)
    td_rms = RunningTDRMS(device, args.td_rms_decay, args.td_rms_min)
    kl_controller = KLController(args)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    global_step = 0
    start_time = time.time()
    trace_decay = args.gamma * args.trace_lambda

    for update in range(1, args.num_updates + 1):
        global_step += args.num_envs
        frac = 1.0 - (update - 1.0) / args.num_updates if args.anneal_lr else 1.0
        obs = next_obs
        with torch.no_grad():
            action, action_z, value, deploy_actor, deploy_critic, alpha_raw, beta_raw, old_dist = agent.act(obs, args)
            old_alpha = old_dist.concentration1.clone()
            old_beta = old_dist.concentration0.clone()

        next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
        bootstrap_np = bootstrap_observations(next_obs_np, truncated_np, infos)
        bootstrap_obs = torch.as_tensor(bootstrap_np, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool, device=device)
        truncated = torch.as_tensor(truncated_np, dtype=torch.bool, device=device)
        done = terminated | truncated
        with torch.no_grad():
            next_value = agent.get_value(bootstrap_obs, args)
            td_error = reward + args.gamma * (~terminated).float() * next_value - value
            actor_delta = td_rms.normalize(td_error, args.actor_td_clip)
            critic_delta = td_error.clamp(-args.critic_td_clip, args.critic_td_clip)

        collect_diagnostics = update == 1 or update % args.log_interval == 0
        actor_states, actor_residuals, actor_diag = agent.settle_actor(
            obs, action_z, args, collect_diagnostics
        )
        critic_states, critic_residuals, critic_diag = agent.settle_critic(
            obs, args, collect_diagnostics
        )
        with torch.no_grad():
            alpha = 1.0 + F.softplus(alpha_raw)
            beta = 1.0 + F.softplus(beta_raw)
            alpha_score, beta_score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, action_z)
            alpha_w, alpha_b = linear_scores(alpha_score, deploy_actor[-1])
            beta_w, beta_b = linear_scores(beta_score, deploy_actor[-1])
            value_w = deploy_critic[-1].unsqueeze(1)
            value_b = torch.ones_like(value_w[:, :, 0])
            actor_local_scores = agent.actor_pc.local_scores(obs, actor_states)
            critic_local_scores = agent.critic_pc.local_scores(obs, critic_states)

            # Crucial ordering: current eligibility enters before this transition's
            # delta is applied; terminal rows reset only after receiving that delta.
            actor_head_trace.accumulate([alpha_w, alpha_b, beta_w, beta_b], trace_decay)
            critic_head_trace.accumulate([value_w, value_b], trace_decay)
            actor_local_trace.accumulate(actor_local_scores, trace_decay)
            critic_local_trace.accumulate(critic_local_scores, trace_decay)
            actor_head_dir = actor_head_trace.modulated_mean(actor_delta)
            actor_local_dir = actor_local_trace.modulated_mean(actor_delta)
            critic_head_dir = [
                args.vf_coef * direction for direction in critic_head_trace.modulated_mean(critic_delta)
            ]
            critic_local_dir = [
                args.vf_coef * direction for direction in critic_local_trace.modulated_mean(critic_delta)
            ]
            actor_head_norm = direction_norm(actor_head_dir)
            actor_local_norm = direction_norm(actor_local_dir)
            critic_head_norm = direction_norm(critic_head_dir)
            critic_local_norm = direction_norm(critic_local_dir)
            actor_head_count = len(actor_head_dir)
            critic_head_count = len(critic_head_dir)
            actor_joint, actor_joint_norm = clip_directions(
                actor_head_dir + actor_local_dir, args.max_direction_norm
            )
            critic_joint, critic_joint_norm = clip_directions(
                critic_head_dir + critic_local_dir, args.max_direction_norm
            )
            actor_head_dir, actor_local_dir = (
                actor_joint[:actor_head_count], actor_joint[actor_head_count:]
            )
            critic_head_dir, critic_local_dir = (
                critic_joint[:critic_head_count], critic_joint[critic_head_count:]
            )

            snapshot = agent.snapshot_actor()
            actor_scale = frac * kl_controller.scale
            apply_head_directions(agent, actor_head_dir, args.actor_learning_rate * actor_scale, actor=True)
            apply_local_directions(agent.actor_pc, actor_local_dir, args.local_learning_rate * actor_scale)
            new_states = agent.deployment(agent.actor_pc, obs, args)
            _, _, new_dist = agent.actor_outputs(new_states[-1])
            proposed_kl = float(kl_divergence(Beta(old_alpha, old_beta), new_dist).sum(1).mean())
            hard_scale = 1.0
            if not math.isfinite(proposed_kl) or proposed_kl > args.max_kl:
                # The hierarchy makes KL nonlinear in the joint head/local step.
                # Bisection gives an actual hard cap instead of relying on a
                # quadratic small-step approximation.
                low, high = 0.0, actor_scale
                for _ in range(10):
                    candidate = 0.5 * (low + high)
                    agent.restore_actor(snapshot)
                    apply_head_directions(agent, actor_head_dir, args.actor_learning_rate * candidate, actor=True)
                    apply_local_directions(agent.actor_pc, actor_local_dir, args.local_learning_rate * candidate)
                    candidate_states = agent.deployment(agent.actor_pc, obs, args)
                    _, _, candidate_dist = agent.actor_outputs(candidate_states[-1])
                    candidate_kl = float(
                        kl_divergence(Beta(old_alpha, old_beta), candidate_dist).sum(1).mean()
                    )
                    if candidate_kl <= args.max_kl:
                        low = candidate
                    else:
                        high = candidate
                agent.restore_actor(snapshot)
                hard_scale = low / max(actor_scale, 1e-12)
                actor_scale = low
                apply_head_directions(agent, actor_head_dir, args.actor_learning_rate * actor_scale, actor=True)
                apply_local_directions(agent.actor_pc, actor_local_dir, args.local_learning_rate * actor_scale)
                new_states = agent.deployment(agent.actor_pc, obs, args)
                _, _, new_dist = agent.actor_outputs(new_states[-1])
            post_kl = float(kl_divergence(Beta(old_alpha, old_beta), new_dist).sum(1).mean())
            kl_controller.adapt(post_kl)

            apply_head_directions(agent, critic_head_dir, args.critic_learning_rate * frac, actor=False)
            apply_local_directions(agent.critic_pc, critic_local_dir, args.local_learning_rate * frac)
            agent.actor_pc.update_precisions(actor_residuals, args)
            agent.critic_pc.update_precisions(critic_residuals, args)
            for trace in traces:
                trace.reset(done)

        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = float(np.asarray(info["episode"]["r"]))
                    episodic_length = int(np.asarray(info["episode"]["l"]))
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        if update == 1 or update % args.log_interval == 0:
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/learning_rate_fraction", frac, global_step)
            writer.add_scalar("td/error_mean", td_error.mean(), global_step)
            writer.add_scalar("td/error_rms", td_rms.mean_square.sqrt(), global_step)
            writer.add_scalar("trust/post_update_kl", post_kl, global_step)
            writer.add_scalar("trust/proposed_kl", proposed_kl, global_step)
            writer.add_scalar("trust/controller_scale", kl_controller.scale, global_step)
            writer.add_scalar("trust/hard_scale", hard_scale, global_step)
            writer.add_scalar("trust/accepted_actor_scale", actor_scale, global_step)
            writer.add_scalar("direction/actor_head", actor_head_norm, global_step)
            writer.add_scalar("direction/critic_head", critic_head_norm, global_step)
            writer.add_scalar("direction/actor_local", actor_local_norm, global_step)
            writer.add_scalar("direction/critic_local", critic_local_norm, global_step)
            writer.add_scalar("direction/actor_joint", actor_joint_norm, global_step)
            writer.add_scalar("direction/critic_joint", critic_joint_norm, global_step)
            writer.add_scalar("pc/actor_steps", actor_diag["steps"], global_step)
            writer.add_scalar("pc/critic_steps", critic_diag["steps"], global_step)
            writer.add_scalar("pc/actor_inference_step_mean", actor_diag["mean_step"], global_step)
            writer.add_scalar("pc/critic_inference_step_mean", critic_diag["mean_step"], global_step)
            writer.add_scalar("pc/actor_max_curvature", actor_diag["max_curvature"], global_step)
            writer.add_scalar("pc/critic_max_curvature", critic_diag["max_curvature"], global_step)
            actor_trace_rms = actor_local_trace.layer_rms()
            critic_trace_rms = critic_local_trace.layer_rms()
            for idx, edge in enumerate(agent.actor_pc.edges):
                writer.add_scalar(f"pc_actor/layer_{idx}_displacement", actor_diag["displacements"][idx].mean(), global_step)
                writer.add_scalar(f"pc_actor/layer_{idx}_residual", actor_diag["residuals"][idx].mean(), global_step)
                writer.add_scalar(f"pc_actor/layer_{idx}_trace_rms", actor_trace_rms[idx], global_step)
                writer.add_scalar(
                    f"pc_actor/layer_{idx}_update_rms",
                    actor_local_dir[idx].square().mean().sqrt() * args.local_learning_rate * actor_scale,
                    global_step,
                )
                writer.add_scalar(f"pc_actor/layer_{idx}_precision_mean", edge.precision.mean(), global_step)
                writer.add_scalar(f"pc_actor/layer_{idx}_precision_max", edge.precision.max(), global_step)
            for idx, edge in enumerate(agent.critic_pc.edges):
                writer.add_scalar(f"pc_critic/layer_{idx}_displacement", critic_diag["displacements"][idx].mean(), global_step)
                writer.add_scalar(f"pc_critic/layer_{idx}_residual", critic_diag["residuals"][idx].mean(), global_step)
                writer.add_scalar(f"pc_critic/layer_{idx}_trace_rms", critic_trace_rms[idx], global_step)
                writer.add_scalar(
                    f"pc_critic/layer_{idx}_update_rms",
                    critic_local_dir[idx].square().mean().sqrt()
                    * args.local_learning_rate * frac,
                    global_step,
                )
                writer.add_scalar(f"pc_critic/layer_{idx}_precision_mean", edge.precision.mean(), global_step)
                writer.add_scalar(f"pc_critic/layer_{idx}_precision_max", edge.precision.max(), global_step)
            print(f"update={update}, global_step={global_step}, SPS={sps}, td_rms={float(td_rms.mean_square.sqrt()):.3f}, kl={post_kl:.6f}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
