# Predictive-coding actor-critic with momentum response dynamics and TD(lambda), v5.
#
# Every vector transition updates immediately: there is no rollout buffer, GAE,
# PPO objective, replay, or backpropagation through the hidden hierarchy. Actor
# inference injects the exact frozen score of a smoothly concentration-bounded
# Beta likelihood. Fresh linear-response neurons then evolve under simultaneous,
# local, block-normalized heavy-ball dynamics around each transition's free
# state. Their equilibrium is the exact first-order PC response, without a Fisher
# spring, finite nudge, layerwise sweep, or cross-transition state persistence.
# The critic uses the analogous unit output score. Local response residuals enter
# exact per-environment, per-parameter TD(lambda) traces and streaming Adam. This
# is prospective predictive coding plus temporal eligibility, not backpropagation
# or a conjugate Bayesian posterior update.
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
    anneal_lr: bool = True
    vf_coef: float = 0.5
    td_rms_decay: float = 0.999
    td_rms_min: float = 0.1
    actor_td_clip: float = 10.0
    critic_td_clip: float = 10.0
    log_interval: int = 100

    hidden_size: int = 64
    pc_num_hidden_layers: int = 6
    pc_dynamics_steps: int = 16
    pc_dynamics_step_size: float = 1.4
    pc_dynamics_momentum: float = 0.82
    pc_dynamics_block_damping: float = 0.05
    pc_input_activation: str = "identity"
    pc_hidden_activation: str = "tanh"
    beta_edge_offset: float = 1.0
    beta_concentration_min: float = 1.0
    beta_concentration_max: float = 32.0

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


class RunningTDRMS:
    def __init__(self, device, decay, minimum):
        self.decay = decay
        self.minimum = minimum
        self.mean_square = torch.ones((), device=device)
        self.initialized = torch.zeros((), dtype=torch.bool, device=device)

    @torch.no_grad()
    def normalize(self, delta, clip):
        finite = torch.isfinite(delta)
        square = torch.where(finite, delta, 0.0).square().sum() / finite.sum().clamp_min(1)
        candidate = torch.where(
            self.initialized,
            torch.lerp(self.mean_square, square, 1.0 - self.decay),
            square,
        )
        has_finite = finite.any()
        self.mean_square.copy_(torch.where(has_finite, candidate, self.mean_square))
        self.initialized.logical_or_(has_finite)
        normalized = torch.where(
            finite, delta / self.mean_square.sqrt().clamp_min(self.minimum), 0.0
        )
        return normalized.clamp(-clip, clip) if clip > 0 else normalized


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


def bounded_beta_parameters(raw, edge_offset, concentration_min, concentration_max):
    """Map mean/precision coordinates to a smooth, finite-concentration Beta."""
    if edge_offset <= 0:
        raise ValueError("beta_edge_offset must be positive")
    if concentration_min < 0 or concentration_max <= concentration_min:
        raise ValueError("require 0 <= beta_concentration_min < beta_concentration_max")
    mean_logit, concentration_raw = raw.chunk(2, dim=1)
    allocation = torch.sigmoid(mean_logit)
    concentration_free = F.softplus(concentration_raw)
    concentration_range = concentration_max - concentration_min
    concentration = concentration_min + concentration_range * concentration_free / (
        concentration_range + concentration_free
    )
    alpha = edge_offset + allocation * concentration
    beta = edge_offset + (1.0 - allocation) * concentration
    return alpha, beta, allocation, concentration


def bounded_beta_score(
    raw, alpha, beta, allocation, concentration, action_z, concentration_min, concentration_max
):
    """Exact action log-likelihood score in mean/precision raw coordinates."""
    _, concentration_raw = raw.chunk(2, dim=1)
    z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    common = torch.digamma(alpha + beta)
    alpha_score = z.log() - torch.digamma(alpha) + common
    beta_score = (1.0 - z).log() - torch.digamma(beta) + common

    allocation_derivative = allocation * (1.0 - allocation)
    mean_score = concentration * allocation_derivative * (alpha_score - beta_score)
    concentration_free = F.softplus(concentration_raw)
    concentration_range = concentration_max - concentration_min
    concentration_derivative = (
        concentration_range**2
        * torch.sigmoid(concentration_raw)
        / (concentration_range + concentration_free).square()
    )
    precision_score = concentration_derivative * (
        allocation * alpha_score + (1.0 - allocation) * beta_score
    )
    return torch.cat([mean_score, precision_score], dim=1)


def linearized_residuals(responses, weights, derivatives):
    """First-order edge residuals around a zero-error feedforward state."""
    residuals = [responses[0]]
    for layer_idx in range(1, len(responses)):
        predicted_response = F.linear(
            derivatives[layer_idx - 1] * responses[layer_idx - 1],
            weights[layer_idx],
        )
        residuals.append(responses[layer_idx] - predicted_response)
    return residuals


def linearized_gradients(
    residuals, weights, derivatives, output_weight, terminal_score
):
    """Local gradient of the Schur-eliminated quadratic response energy."""
    gradients = []
    for layer_idx, residual in enumerate(residuals):
        gradient = residual
        if layer_idx + 1 < len(residuals):
            correction = F.linear(residuals[layer_idx + 1], weights[layer_idx + 1].T)
            gradient = gradient - derivatives[layer_idx] * correction
        else:
            gradient = gradient - F.linear(terminal_score, output_weight.T)
        gradients.append(gradient)
    return gradients


def make_compiled_dynamics_core(args):
    num_steps = args.pc_dynamics_steps
    step_size = args.pc_dynamics_step_size
    momentum = args.pc_dynamics_momentum

    def core(
        free_states,
        weights,
        derivatives,
        factors,
        output_weight,
        terminal_score,
    ):
        responses = [torch.zeros_like(state) for state in free_states]
        velocities = [torch.zeros_like(state) for state in free_states]
        for _ in range(num_steps):
            residuals = linearized_residuals(responses, weights, derivatives)
            gradients = linearized_gradients(
                residuals, weights, derivatives, output_weight, terminal_score
            )
            velocities = [
                momentum * velocity
                - step_size
                * torch.cholesky_solve(gradient.unsqueeze(2), factor).squeeze(2)
                for velocity, gradient, factor in zip(velocities, gradients, factors)
            ]
            responses = [
                response + velocity
                for response, velocity in zip(responses, velocities)
            ]
        return tuple(responses), tuple(velocities)

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
    def response_geometry(self, free_states, args):
        """Local Jacobians and block gains of the linear response system."""
        derivatives = [
            activation_derivative(free_states[layer_idx], self.edges[layer_idx + 1].activation_name)
            for layer_idx in range(len(self.edges) - 1)
        ]
        batch, hidden = free_states[0].shape
        eye = torch.eye(hidden, device=free_states[0].device, dtype=free_states[0].dtype)
        factors = []
        for layer_idx, state in enumerate(free_states):
            if layer_idx + 1 < len(self.edges):
                jacobian = self.edges[layer_idx + 1].weight.unsqueeze(0) * derivatives[
                    layer_idx
                ].unsqueeze(1)
                block = jacobian.transpose(1, 2) @ jacobian
            else:
                block = eye.new_zeros(batch, hidden, hidden)
            block = block + (1.0 + args.pc_dynamics_block_damping) * eye
            factor, _ = torch.linalg.cholesky_ex(block, check_errors=False)
            factors.append(factor)
        return derivatives, factors

    @torch.no_grad()
    def response(
        self,
        x,
        output_edge,
        terminal_score,
        args,
        compiled_core=None,
        collect_diagnostics=True,
        free_states=None,
    ):
        if free_states is None:
            free_states = self.initial_states(x)
        derivatives, factors = self.response_geometry(free_states, args)
        weights = tuple(edge.weight for edge in self.edges)
        if compiled_core is not None:
            responses, velocities = compiled_core(
                tuple(free_states),
                weights,
                tuple(derivatives),
                tuple(factors),
                output_edge.weight,
                terminal_score,
            )
            responses, velocities = list(responses), list(velocities)
        else:
            responses = [torch.zeros_like(state) for state in free_states]
            velocities = [torch.zeros_like(state) for state in free_states]
            for _ in range(args.pc_dynamics_steps):
                response_residuals = linearized_residuals(responses, weights, derivatives)
                gradients = linearized_gradients(
                    response_residuals,
                    weights,
                    derivatives,
                    output_edge.weight,
                    terminal_score,
                )
                velocities = [
                    args.pc_dynamics_momentum * velocity
                    - args.pc_dynamics_step_size
                    * torch.cholesky_solve(gradient.unsqueeze(2), factor).squeeze(2)
                    for velocity, gradient, factor in zip(
                        velocities, gradients, factors
                    )
                ]
                responses = [
                    response + velocity
                    for response, velocity in zip(responses, velocities)
                ]

        response_residuals = linearized_residuals(responses, weights, derivatives)
        gradients = linearized_gradients(
            response_residuals,
            weights,
            derivatives,
            output_edge.weight,
            terminal_score,
        )
        zero = free_states[0].new_zeros(())
        if collect_diagnostics:
            free_residuals = self.residuals(x, free_states)
            free_stack = torch.cat(
                [residual.flatten(1) for residual in free_residuals], dim=1
            )
            free_state_stack = torch.cat(
                [state.flatten(1) for state in free_states], dim=1
            )
            response_stack = torch.cat(
                [response.flatten(1) for response in responses], dim=1
            )
            velocity_stack = torch.cat(
                [velocity.flatten(1) for velocity in velocities], dim=1
            )
            residual_stack = torch.cat(
                [residual.flatten(1) for residual in response_residuals], dim=1
            )
            gradient_stack = torch.cat(
                [gradient.flatten(1) for gradient in gradients], dim=1
            )
            terminal_force = F.linear(terminal_score, output_edge.weight.T)
            initial_gradient_rms = terminal_force.square().sum().div(
                gradient_stack.numel()
            ).sqrt()
            energy = sum(
                0.5 * residual.square().sum(dim=1)
                for residual in response_residuals
            ) - (
                terminal_score * F.linear(responses[-1], output_edge.weight)
            ).sum(dim=1)
        diagnostics = {
            "steps": args.pc_dynamics_steps,
            "free_residual_rms": (
                free_stack.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "free_residual_max": (
                free_stack.abs().max() if collect_diagnostics else zero
            ),
            "free_state_rms": (
                free_state_stack.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "free_state_max": (
                free_state_stack.abs().max() if collect_diagnostics else zero
            ),
            "response_rms": (
                response_stack.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "response_max": (
                response_stack.abs().max() if collect_diagnostics else zero
            ),
            "velocity_rms": (
                velocity_stack.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "residual_rms": (
                residual_stack.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "gradient_rms": (
                gradient_stack.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "initial_gradient_rms": initial_gradient_rms if collect_diagnostics else zero,
            "convergence_ratio": (
                gradient_stack.square().mean().sqrt()
                / initial_gradient_rms.clamp_min(1e-12)
                if collect_diagnostics
                else zero
            ),
            "energy_mean": energy.mean() if collect_diagnostics else zero,
            "terminal_score_rms": (
                terminal_score.square().mean().sqrt() if collect_diagnostics else zero
            ),
            "responses": (
                [response.norm(dim=1) for response in responses]
                if collect_diagnostics
                else []
            ),
            "residuals": (
                [residual.norm(dim=1) for residual in response_residuals]
                if collect_diagnostics
                else []
            ),
        }
        return free_states, response_residuals, diagnostics

    @torch.no_grad()
    def local_scores(self, x, free_states, response_residuals):
        scores = []
        source = x
        for layer_idx, (edge, residual) in enumerate(
            zip(self.edges, response_residuals)
        ):
            scores.append(
                residual.unsqueeze(2)
                * edge.augmented_features(source).unsqueeze(1)
            )
            source = free_states[layer_idx]
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


class AugmentedAdam:
    """Adam ascent on local augmented [weight | bias] directions."""

    def __init__(self, edges, beta1, beta2, epsilon):
        self.edges = list(edges)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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
            edge.weight.add_(normalized[:, :-1], alpha=learning_rate)
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


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.actor_output = OutputPredictor(args.hidden_size, 2 * act_dim, std=0.01)
        self.critic_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.critic_output = OutputPredictor(args.hidden_size, 1, std=1.0)
        self.beta_edge_offset = args.beta_edge_offset
        self.beta_concentration_min = args.beta_concentration_min
        self.beta_concentration_max = args.beta_concentration_max
        self._compiled_actor_dynamics = None
        self._compiled_critic_dynamics = None
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
            self._compiled_actor_dynamics = make_compiled_dynamics_core(args)
            self._compiled_critic_dynamics = make_compiled_dynamics_core(args)

    def actor_distribution(self, hidden):
        raw = self.actor_output(hidden)
        alpha, beta, allocation, concentration = bounded_beta_parameters(
            raw,
            self.beta_edge_offset,
            self.beta_concentration_min,
            self.beta_concentration_max,
        )
        return raw, allocation, concentration, Beta(alpha, beta)

    @torch.no_grad()
    def actor_terminal_score(self, hidden, action_z, args):
        output, allocation, concentration, dist = self.actor_distribution(hidden)
        alpha, beta = dist.concentration1, dist.concentration0
        return bounded_beta_score(
            output,
            alpha,
            beta,
            allocation,
            concentration,
            action_z,
            args.beta_concentration_min,
            args.beta_concentration_max,
        )

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
        score = self.actor_terminal_score(free_states[-1], action_z, args)
        free_states, residuals, diagnostics = self.actor_pc.response(
            x,
            self.actor_output,
            score,
            args,
            self._compiled_actor_dynamics,
            collect_diagnostics,
            free_states=free_states,
        )
        output_score = score.unsqueeze(2) * self.actor_output.augmented_features(
            free_states[-1]
        ).unsqueeze(1)
        return free_states, residuals, output_score, diagnostics

    @torch.no_grad()
    def settle_critic(self, x, args, collect_diagnostics=True):
        free_states = self.critic_pc.initial_states(x)
        score = self.critic_output(free_states[-1]).new_ones(
            free_states[-1].shape[0], 1
        )
        free_states, residuals, diagnostics = self.critic_pc.response(
            x,
            self.critic_output,
            score,
            args,
            self._compiled_critic_dynamics,
            collect_diagnostics,
            free_states=free_states,
        )
        output_score = score.unsqueeze(2) * self.critic_output.augmented_features(
            free_states[-1]
        ).unsqueeze(1)
        return free_states, residuals, output_score, diagnostics


def make_trace_banks(agent, args, device):
    batch = args.num_envs
    actor = TraceBank([
        torch.empty(batch, edge.weight.shape[0], edge.weight.shape[1] + 1, device=device)
        for edge in agent.actor_edges()
    ])
    critic = TraceBank([
        torch.empty(batch, edge.weight.shape[0], edge.weight.shape[1] + 1, device=device)
        for edge in agent.critic_edges()
    ])
    return actor, critic


def main():
    args = tyro.cli(Args)
    args.num_updates = args.total_timesteps // args.num_envs
    assert args.cuda and torch.cuda.is_available(), "CUDA is required for this research variant"
    if args.pc_dynamics_steps < 1:
        raise ValueError("pc_dynamics_steps must be positive")
    if args.pc_dynamics_step_size <= 0:
        raise ValueError("pc_dynamics_step_size must be positive")
    if not 0 <= args.pc_dynamics_momentum < 1:
        raise ValueError("pc_dynamics_momentum must be in [0, 1)")
    if args.pc_dynamics_block_damping < 0:
        raise ValueError("pc_dynamics_block_damping must be nonnegative")
    if args.beta_edge_offset <= 0:
        raise ValueError("beta_edge_offset must be positive")
    if args.beta_concentration_min < 0 or args.beta_concentration_max <= args.beta_concentration_min:
        raise ValueError("require 0 <= beta_concentration_min < beta_concentration_max")
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
    actor_trace, critic_trace = make_trace_banks(agent, args, device)
    actor_optimizer = AugmentedAdam(
        agent.actor_edges(), args.adam_beta1, args.adam_beta2, args.adam_epsilon
    )
    critic_optimizer = AugmentedAdam(
        agent.critic_edges(), args.adam_beta1, args.adam_beta2, args.adam_epsilon
    )
    td_rms = RunningTDRMS(device, args.td_rms_decay, args.td_rms_min)

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
            actor_delta = td_rms.normalize(td_error, args.actor_td_clip)
            critic_delta = td_error.clamp(-args.critic_td_clip, args.critic_td_clip)

        collect_diagnostics = update == 1 or update % args.log_interval == 0
        actor_states, actor_residuals, actor_output_score, actor_diag = agent.settle_actor(
            obs, action_z, args, collect_diagnostics
        )
        critic_states, critic_residuals, critic_output_score, critic_diag = agent.settle_critic(
            obs, args, collect_diagnostics
        )
        with torch.no_grad():
            actor_scores = agent.actor_pc.local_scores(
                obs, actor_states, actor_residuals
            )
            actor_scores.append(actor_output_score)
            critic_scores = agent.critic_pc.local_scores(
                obs, critic_states, critic_residuals
            )
            critic_scores.append(critic_output_score)
            actor_valid = finite_example_rows(actor_scores + actor_residuals + [td_error, actor_delta])
            critic_valid = finite_example_rows(critic_scores + critic_residuals + [td_error, critic_delta])
            actor_trace.reset(~actor_valid)
            critic_trace.reset(~critic_valid)
            actor_scores = zero_invalid_rows(actor_scores, actor_valid)
            critic_scores = zero_invalid_rows(critic_scores, critic_valid)
            actor_delta = torch.where(actor_valid, actor_delta, 0.0)
            critic_delta = torch.where(critic_valid, critic_delta, 0.0)

            # Current eligibility enters before its TD error; done rows receive
            # this transition once and are reset only after the parameter update.
            actor_trace.accumulate(actor_scores, trace_decay)
            critic_trace.accumulate(critic_scores, trace_decay)
            actor_directions = actor_trace.modulated_mean(actor_delta)
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
            actor_trace.reset(done)
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
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/learning_rate_fraction", frac, global_step)
            writer.add_scalar("td/error_mean", td_error.mean(), global_step)
            writer.add_scalar("td/error_rms", td_rms.mean_square.sqrt(), global_step)
            writer.add_scalar("td/actor_nonfinite_fraction", (~actor_valid).float().mean(), global_step)
            writer.add_scalar("td/critic_nonfinite_fraction", (~critic_valid).float().mean(), global_step)
            writer.add_scalar("direction/actor", actor_direction_norm, global_step)
            writer.add_scalar("direction/critic", critic_direction_norm, global_step)
            writer.add_scalar("policy/entropy", behavior_dist.entropy().sum(1).mean(), global_step)
            writer.add_scalar("policy/alpha_mean", behavior_dist.concentration1.mean(), global_step)
            writer.add_scalar("policy/alpha_max", behavior_dist.concentration1.max(), global_step)
            writer.add_scalar("policy/beta_mean", behavior_dist.concentration0.mean(), global_step)
            writer.add_scalar("policy/beta_max", behavior_dist.concentration0.max(), global_step)
            writer.add_scalar(
                "policy/concentration_mean",
                (behavior_dist.concentration1 + behavior_dist.concentration0).mean(),
                global_step,
            )
            writer.add_scalar(
                "policy/action_mean_distance_from_center",
                (behavior_dist.mean - 0.5).abs().mean(),
                global_step,
            )
            for name, diagnostics in (("actor", actor_diag), ("critic", critic_diag)):
                writer.add_scalar(f"pc/{name}_steps", diagnostics["steps"], global_step)
                writer.add_scalar(f"pc/{name}_free_residual_rms", diagnostics["free_residual_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_residual_max", diagnostics["free_residual_max"], global_step)
                writer.add_scalar(f"pc/{name}_terminal_score_rms", diagnostics["terminal_score_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_state_rms", diagnostics["free_state_rms"], global_step)
                writer.add_scalar(f"pc/{name}_free_state_max", diagnostics["free_state_max"], global_step)
                writer.add_scalar(f"pc/{name}_response_rms", diagnostics["response_rms"], global_step)
                writer.add_scalar(f"pc/{name}_response_max", diagnostics["response_max"], global_step)
                writer.add_scalar(f"pc/{name}_velocity_rms", diagnostics["velocity_rms"], global_step)
                writer.add_scalar(f"pc/{name}_residual_rms", diagnostics["residual_rms"], global_step)
                writer.add_scalar(f"pc/{name}_gradient_rms", diagnostics["gradient_rms"], global_step)
                writer.add_scalar(f"pc/{name}_initial_gradient_rms", diagnostics["initial_gradient_rms"], global_step)
                writer.add_scalar(f"pc/{name}_convergence_ratio", diagnostics["convergence_ratio"], global_step)
                writer.add_scalar(f"pc/{name}_energy_mean", diagnostics["energy_mean"], global_step)
            actor_trace_rms = actor_trace.layer_rms()
            critic_trace_rms = critic_trace.layer_rms()
            for idx in range(len(agent.actor_edges())):
                writer.add_scalar(f"pc_actor/edge_{idx}_trace_rms", actor_trace_rms[idx], global_step)
                writer.add_scalar(f"pc_actor/edge_{idx}_adam_update_rms", actor_update_rms[idx], global_step)
                writer.add_scalar(f"pc_actor/edge_{idx}_weight_rms", agent.actor_edges()[idx].weight.square().mean().sqrt(), global_step)
                writer.add_scalar(f"pc_actor/edge_{idx}_weight_max", agent.actor_edges()[idx].weight.abs().max(), global_step)
                if idx < len(actor_residuals):
                    writer.add_scalar(f"pc_actor/layer_{idx}_response", actor_diag["responses"][idx].mean(), global_step)
                    writer.add_scalar(f"pc_actor/layer_{idx}_residual", actor_diag["residuals"][idx].mean(), global_step)
            for idx in range(len(agent.critic_edges())):
                writer.add_scalar(f"pc_critic/edge_{idx}_trace_rms", critic_trace_rms[idx], global_step)
                writer.add_scalar(f"pc_critic/edge_{idx}_adam_update_rms", critic_update_rms[idx], global_step)
                writer.add_scalar(f"pc_critic/edge_{idx}_weight_rms", agent.critic_edges()[idx].weight.square().mean().sqrt(), global_step)
                writer.add_scalar(f"pc_critic/edge_{idx}_weight_max", agent.critic_edges()[idx].weight.abs().max(), global_step)
                if idx < len(critic_residuals):
                    writer.add_scalar(f"pc_critic/layer_{idx}_response", critic_diag["responses"][idx].mean(), global_step)
                    writer.add_scalar(f"pc_critic/layer_{idx}_residual", critic_diag["residuals"][idx].mean(), global_step)
            print(
                f"update={update}, global_step={global_step}, SPS={sps}, "
                f"td_rms={float(td_rms.mean_square.sqrt()):.3f}, "
                f"actor_pc_stationarity={float(actor_diag['convergence_ratio']):.3f}"
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
