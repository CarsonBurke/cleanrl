# Fisher-clamped predictive coding with streaming advantage normalization, v8.
#
# Every vector transition updates immediately: there is no rollout buffer, GAE,
# PPO objective, replay, or backpropagation through the hidden hierarchy. Actor
# inference uses an explicit quadratic output node whose frozen per-example metric
# is the raw-parameter Beta Fisher and whose target is one natural-score nudge.
# The critic uses the corresponding unit-metric value nudge. Ten reverse block
# Gauss-Seidel sweeps propagate those output errors through unconstrained Gaussian
# hidden states; all local scores are divided by the nudge before entering exact
# per-environment, per-parameter TD(lambda) traces. Two actor traces assign every
# score lagged lambda-return moments without rollout storage or current-sample
# leakage; the critic retains raw TD(lambda). Directions use streaming AdamW with
# no decay by default. This is prospective predictive coding plus temporal
# eligibility, not a conjugate Bayesian posterior update.
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
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    anneal_lr: bool = True
    vf_coef: float = 0.5
    advnorm_decay: float = 0.999
    advnorm_std_min: float = 0.1
    advnorm_stats_winsor: float = 20.0
    actor_td_clip: float = 10.0
    critic_td_clip: float = 10.0
    log_interval: int = 100

    hidden_size: int = 64
    pc_num_hidden_layers: int = 6
    pc_inference_steps: int = 10
    pc_inference_scale: float = 1.0
    pc_actor_nudge: float = 0.05
    pc_critic_nudge: float = 0.05
    pc_fisher_damping: float = 1e-3
    pc_curvature_damping: float = 0.05
    pc_input_activation: str = "identity"
    pc_hidden_activation: str = "tanh"

    compile: bool = False
    compile_mode: Optional[str] = "reduce-overhead"
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


def bootstrap_observations(next_obs, truncations, infos):
    """Use final observations for time limits; autoreset observations are wrong."""
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


class StreamingAdvantageMoments:
    """Lagged moments of a causal proxy for the forward lambda-return."""

    def __init__(self, device, num_envs, decay, minimum, stats_winsor):
        self.decay = decay
        self.minimum = minimum
        self.stats_winsor = stats_winsor
        self.mean = torch.zeros((), device=device)
        self.variance = torch.ones((), device=device)
        self.proxy = torch.zeros(num_envs, device=device)
        self.initialized = torch.zeros((), dtype=torch.bool, device=device)

    @torch.no_grad()
    def assign_and_update(self, delta, done, trace_decay):
        if delta.shape != self.proxy.shape:
            raise ValueError("delta shape must match num_envs")
        finite = torch.isfinite(delta)
        has_finite = finite.any()
        assigned_mean = torch.where(self.initialized, self.mean, torch.zeros_like(self.mean))
        assigned_std = torch.where(
            self.initialized,
            self.variance.sqrt().clamp_min(self.minimum),
            torch.ones_like(self.variance),
        )

        next_proxy = torch.where(finite, delta + trace_decay * self.proxy, 0.0)
        stats_proxy = next_proxy
        if self.stats_winsor > 0:
            radius = self.stats_winsor * assigned_std
            stats_proxy = torch.where(
                self.initialized,
                next_proxy.clamp(assigned_mean - radius, assigned_mean + radius),
                next_proxy,
            )
        count = finite.sum().clamp_min(1)
        batch_mean = torch.where(finite, stats_proxy, 0.0).sum() / count
        batch_variance = torch.where(
            finite, (stats_proxy - batch_mean).square(), 0.0
        ).sum() / count
        rate = 1.0 - self.decay
        difference = batch_mean - self.mean
        updated_mean = self.mean + rate * difference
        updated_variance = (
            self.decay * self.variance
            + rate * batch_variance
            + self.decay * rate * difference.square()
        )
        self.mean.copy_(
            torch.where(
                has_finite,
                torch.where(self.initialized, updated_mean, batch_mean),
                self.mean,
            )
        )
        self.variance.copy_(
            torch.where(
                has_finite,
                torch.where(self.initialized, updated_variance, batch_variance).clamp_min(0.0),
                self.variance,
            )
        )
        self.initialized.logical_or_(has_finite)
        self.proxy.copy_(torch.where(done | ~finite, torch.zeros_like(next_proxy), next_proxy))
        return assigned_mean, assigned_std


class LocalPredictor(nn.Module):
    """Fixed-unit-precision Gaussian edge with an augmented local parameter."""

    def __init__(self, in_dim, out_dim, activation_name, std=math.sqrt(2)):
        super().__init__()
        linear = layer_init(nn.Linear(in_dim, out_dim), std=std)
        self.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
        self.activation_name = activation_name

    def features(self, source):
        return activation(source, self.activation_name)

    def augmented_features(self, source):
        features = self.features(source)
        return torch.cat([features, torch.ones_like(features[:, :1])], dim=1)

    def forward(self, source):
        return F.linear(self.features(source), self.weight, self.bias)

    def residual(self, source, target):
        return target - self(source)


class OutputPredictor(LocalPredictor):
    def __init__(self, in_dim, out_dim, std):
        super().__init__(in_dim, out_dim, "identity", std=std)


def beta_head_scores(alpha_raw, beta_raw, alpha, beta, action_z):
    """Exact score of a sampled action with respect to raw Beta outputs."""
    z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    common = torch.digamma(alpha + beta)
    alpha_score = (z.log() - torch.digamma(alpha) + common) * torch.sigmoid(alpha_raw)
    beta_score = ((1.0 - z).log() - torch.digamma(beta) + common) * torch.sigmoid(beta_raw)
    return torch.cat([alpha_score, beta_score], dim=1)


def beta_fisher_metric(alpha_raw, beta_raw, alpha, beta, damping):
    """Per-example raw-coordinate Beta Fisher, block diagonal over actions."""
    if damping <= 0:
        raise ValueError("pc_fisher_damping must be positive")
    common = torch.polygamma(1, alpha + beta)
    sigma_alpha = torch.sigmoid(alpha_raw)
    sigma_beta = torch.sigmoid(beta_raw)
    faa = (torch.polygamma(1, alpha) - common) * sigma_alpha.square()
    fbb = (torch.polygamma(1, beta) - common) * sigma_beta.square()
    fab = -common * sigma_alpha * sigma_beta
    batch, action_dim = alpha.shape
    metric = alpha.new_zeros(batch, 2 * action_dim, 2 * action_dim)
    idx = torch.arange(action_dim, device=alpha.device)
    metric[:, idx, idx] = faa
    metric[:, action_dim + idx, action_dim + idx] = fbb
    metric[:, idx, action_dim + idx] = fab
    metric[:, action_dim + idx, idx] = fab
    metric.diagonal(dim1=-2, dim2=-1).add_(damping)
    return metric


def natural_score_target(output, metric, score, nudge):
    """Freeze q_bar = q_0 + beta P^-1 score for a quadratic output node."""
    if nudge <= 0:
        raise ValueError("PC nudges must be positive")
    natural_score = torch.linalg.solve(metric, score.unsqueeze(-1)).squeeze(-1)
    return output + nudge * natural_score


def metric_times(metric, vector):
    return torch.bmm(metric, vector.unsqueeze(-1)).squeeze(-1)


def make_compiled_settle_core(args, activation_names):
    num_layers = len(activation_names)
    num_steps = args.pc_inference_steps
    scale = args.pc_inference_scale

    def core(x, initial_states, weights, biases, factors, output_weight, output_bias, target, metric):
        states = list(initial_states)
        for _ in range(num_steps):
            for layer_idx in reversed(range(num_layers)):
                source = x if layer_idx == 0 else states[layer_idx - 1]
                mean = F.linear(activation(source, activation_names[layer_idx]), weights[layer_idx], biases[layer_idx])
                gradient = states[layer_idx] - mean
                if layer_idx + 1 < num_layers:
                    downstream_mean = F.linear(
                        activation(states[layer_idx], activation_names[layer_idx + 1]),
                        weights[layer_idx + 1],
                        biases[layer_idx + 1],
                    )
                    downstream_error = states[layer_idx + 1] - downstream_mean
                    correction = F.linear(downstream_error, weights[layer_idx + 1].T)
                    gradient = gradient - activation_derivative(
                        states[layer_idx], activation_names[layer_idx + 1]
                    ) * correction
                else:
                    output_error = target - F.linear(states[layer_idx], output_weight, output_bias)
                    output_force = metric_times(metric, output_error)
                    gradient = gradient - F.linear(output_force, output_weight.T)
                block_step = torch.cholesky_solve(gradient.T, factors[layer_idx]).T
                states[layer_idx] = states[layer_idx] - scale * block_step
        return tuple(states)

    import torch._inductor as inductor

    options = dict(inductor.list_mode_options(args.compile_mode, dynamic=False))
    options["triton.cudagraphs"] = False
    return torch.compile(core, dynamic=False, fullgraph=True, options=options)


class PCHierarchy(nn.Module):
    def __init__(self, input_dim, hidden_size, args):
        super().__init__()
        if args.pc_num_hidden_layers < 1:
            raise ValueError("pc_num_hidden_layers must be positive")
        edges = [LocalPredictor(input_dim, hidden_size, args.pc_input_activation)]
        edges.extend(
            LocalPredictor(hidden_size, hidden_size, args.pc_hidden_activation)
            for _ in range(1, args.pc_num_hidden_layers)
        )
        self.edges = nn.ModuleList(edges)

    def activation_names(self):
        return tuple(edge.activation_name for edge in self.edges)

    @torch.no_grad()
    def initial_states(self, x):
        states = []
        source = x
        for edge in self.edges:
            state = edge(source)
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

    @torch.no_grad()
    def curvature_factors(self, states, output_edge, output_metric, args, collect_diagnostics=True):
        """Exact shared Gauss-Newton hidden blocks at the free states."""
        blocks = []
        hidden = states[0].shape[1]
        eye = torch.eye(hidden, device=states[0].device, dtype=states[0].dtype)
        for layer_idx in range(len(self.edges)):
            block = eye
            if layer_idx + 1 < len(self.edges):
                downstream = self.edges[layer_idx + 1]
                derivatives = activation_derivative(states[layer_idx], downstream.activation_name)
                derivative_gram = derivatives.T @ derivatives / derivatives.shape[0]
                block = block + (downstream.weight.T @ downstream.weight) * derivative_gram
            else:
                mean_metric = output_metric.mean(dim=0)
                block = block + output_edge.weight.T @ mean_metric @ output_edge.weight
            blocks.append(block + args.pc_curvature_damping * eye)
        block_batch = torch.stack(blocks)
        # Unit local curvature plus positive damping makes every block SPD by
        # construction. Avoid eigendecompositions and device synchronization on
        # ordinary vector updates; retain spectra only at telemetry intervals.
        # cholesky_ex(check_errors=False) avoids torch.linalg.cholesky's CUDA to
        # CPU error-check synchronization. The info tensor intentionally stays
        # on device and is not branched on in the hot path.
        cholesky, _ = torch.linalg.cholesky_ex(block_batch, check_errors=False)
        if collect_diagnostics:
            eigenvalues = torch.linalg.eigvalsh(block_batch)
            conditions = eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-8)
            max_eigenvalues = eigenvalues[:, -1]
        else:
            conditions = block_batch.new_zeros(block_batch.shape[0])
            max_eigenvalues = block_batch.new_zeros(block_batch.shape[0])
        return cholesky, conditions, max_eigenvalues

    @torch.no_grad()
    def settle(self, x, output_edge, target, output_metric, args, compiled_core=None, collect_diagnostics=True):
        free_states = self.initial_states(x)
        factors, conditions, max_eigenvalues = self.curvature_factors(
            free_states, output_edge, output_metric, args, collect_diagnostics
        )
        if compiled_core is not None:
            states = list(
                compiled_core(
                    x,
                    tuple(free_states),
                    tuple(edge.weight for edge in self.edges),
                    tuple(edge.bias for edge in self.edges),
                    tuple(factors.unbind(0)),
                    output_edge.weight,
                    output_edge.bias,
                    target,
                    output_metric,
                )
            )
        else:
            states = [state.clone() for state in free_states]
            for _ in range(args.pc_inference_steps):
                for layer_idx in reversed(range(len(states))):
                    source = x if layer_idx == 0 else states[layer_idx - 1]
                    edge = self.edges[layer_idx]
                    gradient = states[layer_idx] - edge(source)
                    if layer_idx + 1 < len(states):
                        downstream = self.edges[layer_idx + 1]
                        downstream_error = states[layer_idx + 1] - downstream(states[layer_idx])
                        correction = F.linear(downstream_error, downstream.weight.T)
                        gradient = gradient - activation_derivative(
                            states[layer_idx], downstream.activation_name
                        ) * correction
                    else:
                        output_error = target - output_edge(states[layer_idx])
                        output_force = metric_times(output_metric, output_error)
                        gradient = gradient - F.linear(output_force, output_edge.weight.T)
                    block_step = torch.cholesky_solve(gradient.T, factors[layer_idx]).T
                    states[layer_idx] = states[layer_idx] - args.pc_inference_scale * block_step
        residuals = self.residuals(x, states)
        zero = states[0].new_zeros(())
        if collect_diagnostics:
            free_residuals = self.residuals(x, free_states)
            free_stack = torch.cat([residual.flatten(1) for residual in free_residuals], dim=1)
            free_state_stack = torch.cat([state.flatten(1) for state in free_states], dim=1)
            settled_state_stack = torch.cat([state.flatten(1) for state in states], dim=1)
            settled_means = []
            source = x
            for edge, state in zip(self.edges, states):
                settled_means.append(edge(source))
                source = state
            settled_mean_stack = torch.cat([mean.flatten(1) for mean in settled_means], dim=1)
            residual_stack = torch.cat([residual.flatten(1) for residual in residuals], dim=1)
            free_output = output_edge(free_states[-1])
            settled_output = output_edge(states[-1])
        diagnostics = {
            "steps": args.pc_inference_steps,
            "max_curvature": max_eigenvalues.max() if collect_diagnostics else zero,
            "max_condition": conditions.max() if collect_diagnostics else zero,
            "mean_condition": conditions.mean() if collect_diagnostics else zero,
            "free_residual_rms": free_stack.square().mean().sqrt() if collect_diagnostics else zero,
            "free_residual_max": free_stack.abs().max() if collect_diagnostics else zero,
            "free_state_rms": free_state_stack.square().mean().sqrt() if collect_diagnostics else zero,
            "free_state_max": free_state_stack.abs().max() if collect_diagnostics else zero,
            "settled_state_rms": settled_state_stack.square().mean().sqrt() if collect_diagnostics else zero,
            "settled_state_max": settled_state_stack.abs().max() if collect_diagnostics else zero,
            "settled_predictor_rms": settled_mean_stack.square().mean().sqrt() if collect_diagnostics else zero,
            "settled_predictor_max": settled_mean_stack.abs().max() if collect_diagnostics else zero,
            "residual_rms": residual_stack.square().mean().sqrt() if collect_diagnostics else zero,
            "free_output_rms": free_output.square().mean().sqrt() if collect_diagnostics else zero,
            "free_output_max": free_output.abs().max() if collect_diagnostics else zero,
            "settled_output_rms": settled_output.square().mean().sqrt() if collect_diagnostics else zero,
            "settled_output_max": settled_output.abs().max() if collect_diagnostics else zero,
            "displacements": (
                [(state - free).norm(dim=1) for state, free in zip(states, free_states)]
                if collect_diagnostics
                else []
            ),
            "residuals": (
                [residual.norm(dim=1) for residual in residuals] if collect_diagnostics else []
            ),
        }
        return states, residuals, diagnostics

    @torch.no_grad()
    def local_scores(self, x, states, nudge):
        scores = []
        source = x
        for edge, state in zip(self.edges, states):
            scores.append(
                edge.residual(source, state).unsqueeze(2)
                * edge.augmented_features(source).unsqueeze(1)
                / nudge
            )
            source = state
        return scores


class TraceBank:
    """Independent per-environment, per-parameter accumulating traces."""

    def __init__(self, templates):
        self.traces = [torch.zeros_like(template) for template in templates]

    @torch.no_grad()
    def accumulate(self, scores, decay):
        if len(scores) != len(self.traces):
            raise ValueError("score and trace structures differ")
        for trace, score in zip(self.traces, scores):
            trace.mul_(decay).add_(score)

    @torch.no_grad()
    def modulated_mean(self, delta):
        return [torch.einsum("b,b...->...", delta, trace) / delta.shape[0] for trace in self.traces]

    @torch.no_grad()
    def reset(self, done):
        for trace in self.traces:
            trace[done] = 0

    def layer_rms(self):
        return [trace.square().mean().sqrt() for trace in self.traces]


@torch.no_grad()
def normalized_actor_directions(
    reward_trace, baseline_trace, delta, terminal, trace_decay
):
    """Combine two causal traces into exactly centered lambda-return directions."""
    reward_directions = reward_trace.modulated_mean(delta)
    baseline_coefficient = (1.0 - trace_decay) + trace_decay * terminal.float()
    baseline_directions = baseline_trace.modulated_mean(baseline_coefficient)
    return [
        reward - baseline
        for reward, baseline in zip(reward_directions, baseline_directions)
    ]


class AugmentedAdamW:
    """AdamW ascent on local augmented [weight | bias] directions."""

    def __init__(self, edges, beta1, beta2, epsilon, weight_decay):
        self.edges = list(edges)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.first = [torch.zeros(edge.weight.shape[0], edge.weight.shape[1] + 1,
                                  device=edge.weight.device, dtype=edge.weight.dtype) for edge in self.edges]
        self.second = [torch.zeros_like(moment) for moment in self.first]
        self.step_count = 0

    @torch.no_grad()
    def step(self, directions, learning_rate, collect_diagnostics=False):
        if len(directions) != len(self.edges):
            raise ValueError("direction and optimizer structures differ")
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        update_rms = []
        for edge, direction, first, second in zip(self.edges, directions, self.first, self.second):
            first.mul_(self.beta1).add_(direction, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(direction, direction, value=1.0 - self.beta2)
            normalized = (first / bias_correction1) / (
                (second / bias_correction2).sqrt() + self.epsilon
            )
            # The augmented final column is a bias: it receives its own Adam
            # moments but no decoupled shrinkage.
            decay = 1.0 - learning_rate * self.weight_decay
            edge.weight.mul_(decay).add_(normalized[:, :-1], alpha=learning_rate)
            edge.bias.add_(normalized[:, -1], alpha=learning_rate)
            if collect_diagnostics:
                update_rms.append((learning_rate * normalized).square().mean().sqrt())
        return update_rms


def finite_example_rows(tensors):
    valid = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
    for tensor in tensors:
        valid &= torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
    return valid


def zero_invalid_rows(tensors, valid):
    return [
        torch.where(valid.view(valid.shape[0], *([1] * (tensor.ndim - 1))), tensor, 0.0)
        for tensor in tensors
    ]


def direction_norm(directions):
    return torch.stack([direction.float().square().sum() for direction in directions]).sum().sqrt()


def response_linearity(full_states, half_states, free_states, nudge):
    full = torch.cat(
        [(state - free).flatten(1) for state, free in zip(full_states, free_states)], dim=1
    ) / nudge
    half = torch.cat(
        [(state - free).flatten(1) for state, free in zip(half_states, free_states)], dim=1
    ) / (0.5 * nudge)
    cosine = F.cosine_similarity(full.flatten(), half.flatten(), dim=0)
    relative_error = (full - half).norm() / half.norm().clamp_min(1e-12)
    return full.square().mean().sqrt(), cosine, relative_error


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.actor_output = OutputPredictor(args.hidden_size, 2 * act_dim, std=0.01)
        self.critic_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.critic_output = OutputPredictor(args.hidden_size, 1, std=1.0)
        self._compiled_actor_settle = None
        self._compiled_critic_settle = None
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def actor_edges(self):
        return [*self.actor_pc.edges, self.actor_output]

    def critic_edges(self):
        return [*self.critic_pc.edges, self.critic_output]

    def compile_inference(self, args):
        if args.compile:
            self._compiled_actor_settle = make_compiled_settle_core(args, self.actor_pc.activation_names())
            self._compiled_critic_settle = make_compiled_settle_core(args, self.critic_pc.activation_names())

    def actor_distribution(self, hidden):
        raw = self.actor_output(hidden)
        alpha_raw, beta_raw = raw.chunk(2, dim=1)
        return raw, alpha_raw, beta_raw, Beta(1.0 + F.softplus(alpha_raw), 1.0 + F.softplus(beta_raw))

    @torch.no_grad()
    def actor_target(self, hidden, action_z, args):
        output, alpha_raw, beta_raw, dist = self.actor_distribution(hidden)
        alpha, beta = dist.concentration1, dist.concentration0
        score = beta_head_scores(alpha_raw, beta_raw, alpha, beta, action_z)
        metric = beta_fisher_metric(
            alpha_raw, beta_raw, alpha, beta, args.pc_fisher_damping
        )
        target = natural_score_target(output, metric, score, args.pc_actor_nudge)
        return target, metric, score

    @torch.no_grad()
    def critic_target(self, hidden, args):
        output = self.critic_output(hidden)
        metric = output.new_ones(output.shape[0], 1, 1)
        target = output + args.pc_critic_nudge
        return target, metric

    @torch.no_grad()
    def get_value(self, x):
        return self.critic_output(self.critic_pc.initial_states(x)[-1]).view(-1)

    @torch.no_grad()
    def act(self, x):
        actor_states = self.actor_pc.initial_states(x)
        critic_states = self.critic_pc.initial_states(x)
        _, _, _, dist = self.actor_distribution(actor_states[-1])
        z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        value = self.critic_output(critic_states[-1]).view(-1)
        return action, z, value, actor_states, critic_states, dist

    @torch.no_grad()
    def settle_actor(self, x, action_z, args, collect_diagnostics=True):
        free_states = self.actor_pc.initial_states(x)
        target, metric, score = self.actor_target(free_states[-1], action_z, args)
        states, residuals, diagnostics = self.actor_pc.settle(
            x, self.actor_output, target, metric, args, self._compiled_actor_settle, collect_diagnostics
        )
        if collect_diagnostics:
            free_output = self.actor_output(free_states[-1])
            diagnostics["output_force_error"] = (
                metric_times(metric, target - free_output)
                / args.pc_actor_nudge
                - score
            ).square().mean().sqrt()
            half_states, _, _ = self.actor_pc.settle(
                x,
                self.actor_output,
                torch.lerp(free_output, target, 0.5),
                metric,
                args,
                self._compiled_actor_settle,
                collect_diagnostics=False,
            )
            (
                diagnostics["response_rms"],
                diagnostics["response_linearity_cosine"],
                diagnostics["response_linearity_error"],
            ) = response_linearity(states, half_states, free_states, args.pc_actor_nudge)
        else:
            diagnostics["output_force_error"] = states[0].new_zeros(())
            diagnostics["response_rms"] = states[0].new_zeros(())
            diagnostics["response_linearity_cosine"] = states[0].new_zeros(())
            diagnostics["response_linearity_error"] = states[0].new_zeros(())
        output_error = target - self.actor_output(states[-1])
        output_score = metric_times(metric, output_error).unsqueeze(2) * self.actor_output.augmented_features(states[-1]).unsqueeze(1) / args.pc_actor_nudge
        return states, residuals, output_score, diagnostics

    @torch.no_grad()
    def settle_critic(self, x, args, collect_diagnostics=True):
        free_states = self.critic_pc.initial_states(x)
        target, metric = self.critic_target(free_states[-1], args)
        states, residuals, diagnostics = self.critic_pc.settle(
            x, self.critic_output, target, metric, args, self._compiled_critic_settle, collect_diagnostics
        )
        if collect_diagnostics:
            free_output = self.critic_output(free_states[-1])
            diagnostics["output_force_error"] = (
                metric_times(metric, target - free_output)
                / args.pc_critic_nudge
                - 1.0
            ).square().mean().sqrt()
            half_states, _, _ = self.critic_pc.settle(
                x,
                self.critic_output,
                torch.lerp(free_output, target, 0.5),
                metric,
                args,
                self._compiled_critic_settle,
                collect_diagnostics=False,
            )
            (
                diagnostics["response_rms"],
                diagnostics["response_linearity_cosine"],
                diagnostics["response_linearity_error"],
            ) = response_linearity(states, half_states, free_states, args.pc_critic_nudge)
        else:
            diagnostics["output_force_error"] = states[0].new_zeros(())
            diagnostics["response_rms"] = states[0].new_zeros(())
            diagnostics["response_linearity_cosine"] = states[0].new_zeros(())
            diagnostics["response_linearity_error"] = states[0].new_zeros(())
        output_error = target - self.critic_output(states[-1])
        output_score = metric_times(metric, output_error).unsqueeze(2) * self.critic_output.augmented_features(states[-1]).unsqueeze(1) / args.pc_critic_nudge
        return states, residuals, output_score, diagnostics


def make_trace_banks(agent, args, device):
    batch = args.num_envs
    actor_templates = [
        torch.empty(batch, edge.weight.shape[0], edge.weight.shape[1] + 1, device=device)
        for edge in agent.actor_edges()
    ]
    actor_reward = TraceBank(actor_templates)
    actor_baseline = TraceBank(actor_templates)
    critic = TraceBank([
        torch.empty(batch, edge.weight.shape[0], edge.weight.shape[1] + 1, device=device)
        for edge in agent.critic_edges()
    ])
    return actor_reward, actor_baseline, critic


def main():
    args = tyro.cli(Args)
    args.num_updates = args.total_timesteps // args.num_envs
    assert args.cuda and torch.cuda.is_available(), "CUDA is required for this research variant"
    if args.pc_actor_nudge <= 0 or args.pc_critic_nudge <= 0:
        raise ValueError("PC nudges must be positive")
    if not 0 <= args.advnorm_decay < 1:
        raise ValueError("advnorm_decay must be in [0, 1)")
    if args.advnorm_std_min <= 0:
        raise ValueError("advnorm_std_min must be positive")
    if args.advnorm_stats_winsor < 0:
        raise ValueError("advnorm_stats_winsor must be nonnegative")
    if args.compile:
        print("compile requested; fixed-shape PC inference will be compiled")
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
    actor_reward_trace, actor_baseline_trace, critic_trace = make_trace_banks(
        agent, args, device
    )
    actor_optimizer = AugmentedAdamW(
        agent.actor_edges(), args.adam_beta1, args.adam_beta2, args.adam_epsilon, args.weight_decay
    )
    critic_optimizer = AugmentedAdamW(
        agent.critic_edges(), args.adam_beta1, args.adam_beta2, args.adam_epsilon, args.weight_decay
    )
    advantage_moments = StreamingAdvantageMoments(
        device,
        args.num_envs,
        args.advnorm_decay,
        args.advnorm_std_min,
        args.advnorm_stats_winsor,
    )

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
            action, action_z, value, _, _, behavior_dist = agent.act(obs)
        next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
        bootstrap_np = bootstrap_observations(next_obs_np, truncated_np, infos)
        bootstrap_obs = torch.as_tensor(bootstrap_np, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool, device=device)
        truncated = torch.as_tensor(truncated_np, dtype=torch.bool, device=device)
        done = terminated | truncated
        with torch.no_grad():
            next_value = agent.get_value(bootstrap_obs)
            td_error = reward + args.gamma * (~terminated).float() * next_value - value
            assigned_advantage_mean, assigned_advantage_std = (
                advantage_moments.assign_and_update(td_error, done, trace_decay)
            )
            actor_delta = (
                td_error.clamp(
                    -args.actor_td_clip * assigned_advantage_std,
                    args.actor_td_clip * assigned_advantage_std,
                )
                if args.actor_td_clip > 0
                else td_error
            )
            critic_delta = td_error.clamp(-args.critic_td_clip, args.critic_td_clip)

        collect_diagnostics = update == 1 or update % args.log_interval == 0
        actor_states, actor_residuals, actor_output_score, actor_diag = agent.settle_actor(
            obs, action_z, args, collect_diagnostics
        )
        critic_states, critic_residuals, critic_output_score, critic_diag = agent.settle_critic(
            obs, args, collect_diagnostics
        )
        with torch.no_grad():
            actor_scores = agent.actor_pc.local_scores(obs, actor_states, args.pc_actor_nudge)
            actor_scores.append(actor_output_score)
            critic_scores = agent.critic_pc.local_scores(obs, critic_states, args.pc_critic_nudge)
            critic_scores.append(critic_output_score)
            actor_valid = finite_example_rows(actor_scores + actor_residuals + [td_error, actor_delta])
            critic_valid = finite_example_rows(critic_scores + critic_residuals + [td_error, critic_delta])
            actor_reward_trace.reset(~actor_valid)
            actor_baseline_trace.reset(~actor_valid)
            critic_trace.reset(~critic_valid)
            actor_scores = zero_invalid_rows(actor_scores, actor_valid)
            critic_scores = zero_invalid_rows(critic_scores, critic_valid)
            actor_delta = torch.where(actor_valid, actor_delta, 0.0)
            critic_delta = torch.where(critic_valid, critic_delta, 0.0)

            # Current eligibility enters before its TD error; done rows receive
            # this transition once and are reset only after the parameter update.
            inverse_advantage_std = assigned_advantage_std.reciprocal()
            actor_reward_trace.accumulate(
                [score * inverse_advantage_std for score in actor_scores], trace_decay
            )
            actor_baseline_trace.accumulate(
                [
                    score * (assigned_advantage_mean * inverse_advantage_std)
                    for score in actor_scores
                ],
                trace_decay,
            )
            critic_trace.accumulate(critic_scores, trace_decay)
            actor_directions = normalized_actor_directions(
                actor_reward_trace,
                actor_baseline_trace,
                actor_delta,
                done | (update == args.num_updates),
                trace_decay,
            )
            critic_directions = [
                args.vf_coef * direction for direction in critic_trace.modulated_mean(critic_delta)
            ]
            if collect_diagnostics:
                actor_direction_norm = direction_norm(actor_directions)
                critic_direction_norm = direction_norm(critic_directions)
            actor_update_rms = actor_optimizer.step(
                actor_directions, args.actor_learning_rate * frac, collect_diagnostics
            )
            critic_update_rms = critic_optimizer.step(
                critic_directions, args.critic_learning_rate * frac, collect_diagnostics
            )
            actor_reward_trace.reset(done)
            actor_baseline_trace.reset(done)
            critic_trace.reset(done)

        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = float(np.asarray(info["episode"]["r"]))
                    episodic_length = int(np.asarray(info["episode"]["l"]))
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        if collect_diagnostics:
            sps = int(global_step / (time.time() - start_time))
            running_advantage_std = advantage_moments.variance.sqrt().clamp_min(
                args.advnorm_std_min
            )
            instant_normalized_delta = (
                actor_delta - assigned_advantage_mean
            ) / assigned_advantage_std
            clipped_fraction = (
                (
                    td_error.abs()
                    > args.actor_td_clip * assigned_advantage_std
                )
                .float()
                .mean()
                if args.actor_td_clip > 0
                else torch.zeros((), device=device)
            )
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/learning_rate_fraction", frac, global_step)
            writer.add_scalar("td/error_mean", td_error.mean(), global_step)
            writer.add_scalar("td/error_rms", td_error.square().mean().sqrt(), global_step)
            writer.add_scalar("td/actor_nonfinite_fraction", (~actor_valid).float().mean(), global_step)
            writer.add_scalar("td/critic_nonfinite_fraction", (~critic_valid).float().mean(), global_step)
            writer.add_scalar("advnorm/assigned_mean", assigned_advantage_mean, global_step)
            writer.add_scalar("advnorm/assigned_std", assigned_advantage_std, global_step)
            writer.add_scalar("advnorm/running_mean", advantage_moments.mean, global_step)
            writer.add_scalar("advnorm/running_std", running_advantage_std, global_step)
            writer.add_scalar(
                "advnorm/backward_proxy_rms",
                advantage_moments.proxy.square().mean().sqrt(),
                global_step,
            )
            writer.add_scalar(
                "advnorm/instant_centered_delta_mean",
                instant_normalized_delta.mean(),
                global_step,
            )
            writer.add_scalar(
                "advnorm/instant_centered_delta_rms",
                instant_normalized_delta.square().mean().sqrt(),
                global_step,
            )
            writer.add_scalar("advnorm/actor_delta_clipped_fraction", clipped_fraction, global_step)
            writer.add_scalar("direction/actor", actor_direction_norm, global_step)
            writer.add_scalar("direction/critic", critic_direction_norm, global_step)
            writer.add_scalar("policy/entropy", behavior_dist.entropy().sum(1).mean(), global_step)
            writer.add_scalar("policy/alpha_mean", behavior_dist.concentration1.mean(), global_step)
            writer.add_scalar("policy/alpha_max", behavior_dist.concentration1.max(), global_step)
            writer.add_scalar("policy/beta_mean", behavior_dist.concentration0.mean(), global_step)
            writer.add_scalar("policy/beta_max", behavior_dist.concentration0.max(), global_step)
            for name, diagnostics in (("actor", actor_diag), ("critic", critic_diag)):
                writer.add_scalar(f"pc/{name}_steps", diagnostics["steps"], global_step)
                writer.add_scalar(f"pc/{name}_max_curvature", diagnostics["max_curvature"], global_step)
                writer.add_scalar(f"pc/{name}_max_condition", diagnostics["max_condition"], global_step)
                writer.add_scalar(f"pc/{name}_mean_condition", diagnostics["mean_condition"], global_step)
                writer.add_scalar(f"pc/{name}_free_residual_rms", diagnostics["free_residual_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_residual_max", diagnostics["free_residual_max"], global_step)
                writer.add_scalar(f"pc/{name}_output_force_error", diagnostics["output_force_error"], global_step)
                writer.add_scalar(f"pc/{name}_linear_response_rms", diagnostics["response_rms"], global_step)
                writer.add_scalar(f"pc/{name}_linear_response_cosine", diagnostics["response_linearity_cosine"], global_step)
                writer.add_scalar(f"pc/{name}_linear_response_error", diagnostics["response_linearity_error"], global_step)
                writer.add_scalar(f"pc/{name}_free_state_rms", diagnostics["free_state_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_state_max", diagnostics["free_state_max"], global_step)
                writer.add_scalar(f"pc/{name}_settled_state_rms", diagnostics["settled_state_rms"], global_step)
                writer.add_scalar(f"pc/{name}_settled_state_max", diagnostics["settled_state_max"], global_step)
                writer.add_scalar(f"pc/{name}_settled_predictor_rms", diagnostics["settled_predictor_rms"], global_step)
                writer.add_scalar(f"pc/{name}_settled_predictor_max", diagnostics["settled_predictor_max"], global_step)
                writer.add_scalar(f"pc/{name}_residual_rms", diagnostics["residual_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_output_rms", diagnostics["free_output_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_output_max", diagnostics["free_output_max"], global_step)
                writer.add_scalar(f"pc/{name}_settled_output_rms", diagnostics["settled_output_rms"], global_step)
                writer.add_scalar(f"pc/{name}_settled_output_max", diagnostics["settled_output_max"], global_step)
            actor_reward_trace_rms = actor_reward_trace.layer_rms()
            actor_baseline_trace_rms = actor_baseline_trace.layer_rms()
            critic_trace_rms = critic_trace.layer_rms()
            for idx in range(len(agent.actor_edges())):
                writer.add_scalar(
                    f"pc_actor/edge_{idx}_reward_trace_rms",
                    actor_reward_trace_rms[idx],
                    global_step,
                )
                writer.add_scalar(
                    f"pc_actor/edge_{idx}_baseline_trace_rms",
                    actor_baseline_trace_rms[idx],
                    global_step,
                )
                writer.add_scalar(f"pc_actor/edge_{idx}_adam_update_rms", actor_update_rms[idx], global_step)
                writer.add_scalar(f"pc_actor/edge_{idx}_weight_rms", agent.actor_edges()[idx].weight.square().mean().sqrt(), global_step)
                writer.add_scalar(f"pc_actor/edge_{idx}_weight_max", agent.actor_edges()[idx].weight.abs().max(), global_step)
                if idx < len(actor_residuals):
                    writer.add_scalar(f"pc_actor/layer_{idx}_displacement", actor_diag["displacements"][idx].mean(), global_step)
                    writer.add_scalar(f"pc_actor/layer_{idx}_residual", actor_diag["residuals"][idx].mean(), global_step)
            for idx in range(len(agent.critic_edges())):
                writer.add_scalar(f"pc_critic/edge_{idx}_trace_rms", critic_trace_rms[idx], global_step)
                writer.add_scalar(f"pc_critic/edge_{idx}_adam_update_rms", critic_update_rms[idx], global_step)
                writer.add_scalar(f"pc_critic/edge_{idx}_weight_rms", agent.critic_edges()[idx].weight.square().mean().sqrt(), global_step)
                writer.add_scalar(f"pc_critic/edge_{idx}_weight_max", agent.critic_edges()[idx].weight.abs().max(), global_step)
                if idx < len(critic_residuals):
                    writer.add_scalar(f"pc_critic/layer_{idx}_displacement", critic_diag["displacements"][idx].mean(), global_step)
                    writer.add_scalar(f"pc_critic/layer_{idx}_residual", critic_diag["residuals"][idx].mean(), global_step)
            print(
                f"update={update}, global_step={global_step}, SPS={sps}, "
                f"adv_std={float(running_advantage_std):.3f}, "
                f"actor_pc_response={float(actor_diag['response_rms']):.3f}"
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
