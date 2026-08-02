# Predictive-coding actor-critic with block-curvature inference, v2.
#
# Each vector transition updates immediately: no rollout buffer, GAE, PPO, replay,
# or backpropagation through the hidden hierarchy. Actor and critic hidden states
# settle under the standard Gaussian PC energy, W f(z), using a unit terminal
# direction independent of the current TD error. Five reverse Gauss-Seidel sweeps
# use shared full per-layer Gauss-Newton blocks and cached Cholesky solves, replacing
# v1's fifty scalar-curvature momentum sweeps. Exact head scores and local
# precision-weighted residual scores enter separate per-environment accumulating
# eligibility traces; the subsequent TD error modulates those traces before done
# rows are reset. A recent-state actor KL anchor and a critic value-change guard
# protect streaming updates outside the current behavior slice. This is standard PC
# plus temporal eligibility, not Bayesian PC. Signs follow Appendix C of
# arXiv:2503.24016; its posterior updates are deliberately not mixed into this arm.
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
    actor_learning_rate: float = 3e-3
    critic_learning_rate: float = 3e-3
    local_learning_rate: float = 3e-3
    anneal_lr: bool = True
    vf_coef: float = 0.5
    td_rms_decay: float = 0.999
    td_rms_min: float = 0.1
    actor_td_clip: float = 10.0
    critic_td_clip: float = 10.0
    max_direction_norm: float = 0.5
    max_head_direction_norm: float = 0.25
    max_local_block_direction_norm: float = 0.15
    log_interval: int = 100

    hidden_size: int = 64
    pc_num_hidden_layers: int = 6
    pc_inference_steps: int = 5
    pc_inference_scale: float = 1.0
    """damped block-Newton fraction used by each reverse Gauss-Seidel solve"""
    pc_convergence_tol: float = 0.0
    """optional early-stop tolerance; zero runs fixed max steps without device synchronization"""
    pc_activation_grad_clip: float = 0.0
    """off by default because gradient normalization destroys score magnitude"""
    pc_state_clip: float = 5.0
    pc_actor_terminal_coef: float = 1.0
    pc_critic_terminal_coef: float = 1.0
    pc_compensate_critic_local_score: bool = False
    """normalize critic local scores and precision residuals after a small terminal nudge"""
    pc_input_activation: str = "identity"
    pc_hidden_activation: str = "tanh"
    pc_initial_precision: float = 1.0
    pc_precision_ema: float = 0.001
    pc_precision_ridge: float = 0.01
    pc_precision_min: float = 0.1
    pc_precision_max: float = 10.0
    pc_curvature_damping: float = 0.05
    pc_curvature_refresh_interval: int = 250
    pc_curvature_relative_invalidation: float = 0.005
    """refresh after cumulative accepted downstream-weight motion reaches 0.5%"""
    pc_energy_backtracks: int = 2
    """fixed post-sweep half-step checks; zero disables the descent safeguard"""

    target_kl: float = 0.0001
    max_kl: float = 0.0005
    max_anchor_state_kl: float = 0.005
    kl_adapt_rate: float = 0.05
    kl_scale_min: float = 0.05
    kl_scale_max: float = 4.0
    kl_anchor_capacity: int = 256
    kl_guard_bisection_steps: int = 10

    critic_value_rms_ratio: float = 0.05
    critic_value_rms_floor: float = 0.001
    critic_value_max_change: float = 0.05
    critic_guard_bisection_steps: int = 8

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


def make_compiled_settle_core(args, activation_names, terminal_kind):
    """Build one fixed-shape graph for five block-preconditioned PC sweeps."""
    num_layers = len(activation_names)
    num_steps = args.pc_inference_steps
    inference_scale = args.pc_inference_scale
    grad_clip = args.pc_activation_grad_clip
    state_clip = args.pc_state_clip
    actor_coef = args.pc_actor_terminal_coef
    critic_coef = args.pc_critic_terminal_coef

    def core(x, input_states, weights, biases, precisions, cholesky_factors, terminal_inputs):
        states = list(input_states)
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
                step = torch.cholesky_solve(
                    gradient.unsqueeze(-1),
                    cholesky_factors[layer_idx].expand(gradient.shape[0], -1, -1),
                ).squeeze(-1)
                states[layer_idx] = (states[layer_idx] - inference_scale * step).clamp(
                    -state_clip, state_clip
                )
        return tuple(states)

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
        finite = torch.isfinite(delta)
        has_finite = finite.any()
        square = torch.where(finite, delta, 0.0).square().sum() / finite.sum().clamp_min(1)
        if self.initialized:
            candidate = torch.lerp(self.mean_square, square, 1.0 - self.decay)
        else:
            candidate = square
            self.initialized = True
        self.mean_square.copy_(torch.where(has_finite, candidate, self.mean_square))
        normalized = torch.where(
            finite, delta / self.mean_square.sqrt().clamp_min(self.minimum), 0.0
        )
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
    def update_precision(self, residual, args, valid_rows=None):
        if args.pc_precision_ema <= 0:
            return
        finite_rows = torch.isfinite(residual).all(dim=1)
        if valid_rows is not None:
            finite_rows = finite_rows & valid_rows
        has_valid = finite_rows.any()
        safe_residual = torch.where(finite_rows[:, None], residual.detach(), 0.0)
        batch_second = safe_residual.square().sum(dim=0) / finite_rows.sum().clamp_min(1)
        candidate_second = torch.lerp(
            self.residual_second_moment, batch_second, args.pc_precision_ema
        )
        candidate_precision = (candidate_second + args.pc_precision_ridge).reciprocal().clamp(
            args.pc_precision_min, args.pc_precision_max
        )
        self.residual_second_moment.copy_(
            torch.where(has_valid, candidate_second, self.residual_second_moment)
        )
        self.precision.copy_(torch.where(has_valid, candidate_precision, self.precision))


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
        eye = torch.eye(hidden_size).expand(len(edges), hidden_size, hidden_size).clone()
        self.register_buffer("cached_curvature_blocks", eye.clone())
        self.register_buffer("cached_curvature_cholesky", eye.clone())
        self.register_buffer("cached_curvature_conditions", torch.ones(len(edges)))
        self.register_buffer("cached_curvature_max_eigenvalues", torch.ones(len(edges)))
        self._curvature_initialized = False
        self._curvature_cache_age = args.pc_curvature_refresh_interval
        self._curvature_accumulated_relative_drift = 0.0

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

    def per_example_energy(self, x, states, terminal_energy_fn):
        energy = states[0].new_zeros(states[0].shape[:-1])
        for residual, edge in zip(self.residuals(x, states), self.edges):
            energy = energy + 0.5 * (residual.square() * edge.precision).sum(dim=-1)
        return energy + terminal_energy_fn(states[-1])

    @torch.no_grad()
    def guard_settlement(self, x, proposed_states, terminal_energy_fn, args, diagnostics):
        """Accept the largest half-step interpolation that lowers exact PC energy."""
        deployed = self.initial_states(x, args)
        if args.pc_energy_backtracks <= 0:
            initial_energy = self.per_example_energy(x, deployed, terminal_energy_fn)
            final_energy = self.per_example_energy(x, proposed_states, terminal_energy_fn)
            diagnostics.update(
                {
                    "energy_delta": (final_energy - initial_energy).mean(),
                    "backtrack_scale_mean": initial_energy.new_ones(()),
                    "energy_reject_fraction": initial_energy.new_zeros(()),
                    "state_clip_fraction": torch.stack(
                        [
                            (state.abs() >= 0.999 * args.pc_state_clip).float().mean()
                            for state in proposed_states
                        ]
                    ).mean(),
                }
            )
            return proposed_states, self.residuals(x, proposed_states), diagnostics
        candidate_scales = proposed_states[0].new_tensor(
            [0.5**attempt for attempt in range(args.pc_energy_backtracks)]
        )
        candidate_states = [
            torch.cat(
                [
                    base.unsqueeze(0)
                    + candidate_scales[:, None, None] * (proposal - base).unsqueeze(0),
                    base.unsqueeze(0),
                ],
                dim=0,
            )
            for base, proposal in zip(deployed, proposed_states)
        ]
        all_energy = self.per_example_energy(x, candidate_states, terminal_energy_fn)
        initial_energy = all_energy[-1]
        candidate_energy = all_energy[:-1]
        candidate_states = [states[:-1] for states in candidate_states]
        tolerance = 1e-6 * (1.0 + initial_energy.abs())
        acceptable = torch.isfinite(candidate_energy) & (
            candidate_energy <= initial_energy.unsqueeze(0) + tolerance.unsqueeze(0)
        )
        accepted = acceptable.any(dim=0)
        chosen_indices = acceptable.float().argmax(dim=0)
        batch_indices = torch.arange(initial_energy.shape[0], device=initial_energy.device)
        chosen = [
            torch.where(
                accepted[:, None],
                candidates[chosen_indices, batch_indices],
                base,
            )
            for candidates, base in zip(candidate_states, deployed)
        ]
        chosen_scale = torch.where(accepted, candidate_scales[chosen_indices], 0.0)
        residuals = self.residuals(x, chosen)
        final_energy = torch.where(
            accepted, candidate_energy[chosen_indices, batch_indices], initial_energy
        )
        diagnostics.update(
            {
                "energy_delta": (final_energy - initial_energy).mean(),
                "backtrack_scale_mean": chosen_scale.mean(),
                "energy_reject_fraction": (~accepted).float().mean(),
                "state_clip_fraction": torch.stack(
                    [(state.abs() >= 0.999 * args.pc_state_clip).float().mean() for state in chosen]
                ).mean(),
            }
        )
        if diagnostics["displacements"]:
            diagnostics["displacements"] = [
                (state - base).norm(dim=1) for state, base in zip(chosen, deployed)
            ]
            diagnostics["residuals"] = [residual.norm(dim=1) for residual in residuals]
        return chosen, residuals, diagnostics

    def activation_names(self):
        return tuple(edge.activation_name for edge in self.edges)

    def compiled_settle(self, x, compiled_core, terminal_inputs, args, collect_diagnostics):
        initial_states = tuple(self.initial_states(x, args))
        cholesky, conditions = self.curvature_factors(x, initial_states, args)
        weights = tuple(edge.weight for edge in self.edges)
        biases = tuple(edge.bias for edge in self.edges)
        precisions = tuple(edge.precision for edge in self.edges)
        states = compiled_core(
            x,
            initial_states,
            weights,
            biases,
            precisions,
            tuple(cholesky.unbind(0)),
            terminal_inputs,
        )
        residuals = tuple(self.residuals(x, states))
        diagnostics = {
            "steps": args.pc_inference_steps,
            "mean_step": states[0].new_tensor(args.pc_inference_scale),
            "max_curvature": self.cached_curvature_max_eigenvalues.max(),
            "max_condition": conditions.max(),
            "mean_condition": conditions.mean(),
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
    def curvature_factors(self, x, states, args):
        """Return periodically refreshed shared full SPD block curvatures.

        H_l is the batch-shared mean of
        P_l + D_l W_{l+1}^T P_{l+1} W_{l+1} D_l + damping I.
        Equivalently, its downstream term is the Hadamard product of W^T P W
        and E[d d^T]. All layers have the same hidden width, so their Cholesky
        factorizations are formed in one batched operation and solved per example.
        """
        if self._curvature_initialized:
            self._curvature_cache_age += 1
            if self._curvature_cache_age < args.pc_curvature_refresh_interval:
                return self.cached_curvature_cholesky, self.cached_curvature_conditions
        blocks = []
        eye = torch.eye(states[0].shape[1], device=states[0].device, dtype=states[0].dtype)
        for layer_idx, edge in enumerate(self.edges):
            block = torch.diag(edge.precision)
            if layer_idx + 1 == len(self.edges):
                blocks.append(block + args.pc_curvature_damping * eye)
                continue
            downstream = self.edges[layer_idx + 1]
            derivatives = activation_derivative(states[layer_idx], downstream.activation_name)
            derivative_gram = derivatives.T @ derivatives / derivatives.shape[0]
            weighted = downstream.weight.T @ (
                downstream.precision.unsqueeze(1) * downstream.weight
            )
            block = block + weighted * derivative_gram
            blocks.append(block + args.pc_curvature_damping * eye)
        block_batch = torch.stack(blocks)
        eigenvalues = torch.linalg.eigvalsh(block_batch)
        conditions = eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-8)
        cholesky, info = torch.linalg.cholesky_ex(block_batch)
        if bool(torch.any(info != 0)):
            raise RuntimeError("PC curvature lost positive definiteness")
        self.cached_curvature_blocks.copy_(block_batch)
        self.cached_curvature_cholesky.copy_(cholesky)
        self.cached_curvature_conditions.copy_(conditions)
        self.cached_curvature_max_eigenvalues.copy_(eigenvalues[:, -1])
        self._curvature_initialized = True
        self._curvature_cache_age = 0
        self._curvature_accumulated_relative_drift = 0.0
        return self.cached_curvature_cholesky, self.cached_curvature_conditions

    @torch.no_grad()
    def note_accepted_weight_update(self, directions, learning_rate, args):
        """Invalidate cached blocks after material accepted downstream-weight drift.

        Every edge and bias is included: downstream weights enter blocks directly,
        while earlier parameters move deployed states and therefore derivative
        moments. Summing relative Frobenius step lengths is a conservative path-
        length bound, so oscillating updates cannot hide staleness by cancellation.
        """
        if learning_rate == 0.0 or not directions:
            return
        relative_steps = []
        for edge, direction in zip(self.edges, directions):
            step_norm = abs(learning_rate) * direction.float().norm()
            parameter_norm = (
                edge.weight.float().square().sum() + edge.bias.float().square().sum()
            ).sqrt()
            relative_steps.append(step_norm / parameter_norm.clamp_min(1e-8))
        relative_drift = float(torch.stack(relative_steps).max())
        self._curvature_accumulated_relative_drift += relative_drift
        if self._curvature_accumulated_relative_drift >= args.pc_curvature_relative_invalidation:
            self._curvature_cache_age = args.pc_curvature_refresh_interval

    def settle(self, x, terminal_grad_fn, args, collect_diagnostics=True):
        """Reverse Gauss-Seidel inference with shared full block-Newton solves."""
        states = [state.clone() for state in self.initial_states(x, args)]
        deployed = [state.clone() for state in states]
        cholesky, conditions = self.curvature_factors(x, states, args)
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
                old = states[layer_idx]
                factors = cholesky[layer_idx].expand(grad.shape[0], -1, -1)
                block_step = torch.cholesky_solve(grad.unsqueeze(-1), factors).squeeze(-1)
                states[layer_idx] = (old - args.pc_inference_scale * block_step).clamp(
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
            "mean_step": states[0].new_tensor(args.pc_inference_scale),
            "max_curvature": self.cached_curvature_max_eigenvalues.max(),
            "max_condition": conditions.max(),
            "mean_condition": conditions.mean(),
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
    def update_precisions(self, residuals, args, valid_rows=None):
        """Update from residuals cached before any parameter mutation."""
        assert len(residuals) == len(self.edges)
        for edge, residual in zip(self.edges, residuals):
            edge.update_precision(residual, args, valid_rows)


def beta_head_scores(alpha_raw, beta_raw, alpha, beta, action_z):
    """Exact per-example score with respect to the two unconstrained head outputs."""
    z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    common = torch.digamma(alpha + beta)
    alpha_score = (z.log() - torch.digamma(alpha) + common) * torch.sigmoid(alpha_raw)
    beta_score = ((1.0 - z).log() - torch.digamma(beta) + common) * torch.sigmoid(beta_raw)
    return alpha_score, beta_score


def linear_scores(output_score, features):
    return output_score.unsqueeze(2) * features.unsqueeze(1), output_score


def beta_kl_stats(old_alpha, old_beta, new_dist):
    # Guard arithmetic is cheap at reservoir scale; float64 avoids accepting a
    # cap violation because close Beta distributions lost precision in float32.
    old_dist = Beta(old_alpha.double(), old_beta.double())
    accurate_new_dist = Beta(
        new_dist.concentration1.double(), new_dist.concentration0.double()
    )
    per_state = kl_divergence(old_dist, accurate_new_dist).sum(1)
    return float(per_state.mean()), float(per_state.max())


def finite_example_rows(tensors):
    valid = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
    for tensor in tensors:
        valid = valid & torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
    return valid


def compensate_critic_nudge_tensors(tensors, args):
    """Remove the artificial nudge scale from local scores or PC residuals."""
    if not args.pc_compensate_critic_local_score:
        return tensors
    if args.pc_critic_terminal_coef <= 0:
        raise ValueError("critic score compensation requires a positive terminal coefficient")
    return [tensor / args.pc_critic_terminal_coef for tensor in tensors]


def zero_invalid_rows(tensors, valid):
    return [
        torch.where(valid.view(valid.shape[0], *([1] * (tensor.ndim - 1))), tensor, 0.0)
        for tensor in tensors
    ]


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


def clip_direction_blocks(head_directions, local_directions, args):
    """Cap the head group and every local edge before the final joint cap.

    Uniform scaling within each block keeps every parameter-level score value and
    direction intact, while preventing one unstable edge from consuming the joint
    budget and suppressing the remaining hierarchy.
    """
    clipped_head, _ = clip_directions(head_directions, args.max_head_direction_norm)
    clipped_local = []
    block_cap_active = []
    head_raw_norm = direction_norm(head_directions)
    raw_block_norm_squares = [head_raw_norm.square()]
    block_cap_active.append(
        head_raw_norm > args.max_head_direction_norm
        if args.max_head_direction_norm > 0
        else torch.zeros_like(head_raw_norm, dtype=torch.bool)
    )
    for direction in local_directions:
        raw_norm = direction.float().norm()
        clipped, _ = clip_directions([direction], args.max_local_block_direction_norm)
        clipped_local.extend(clipped)
        raw_block_norm_squares.append(raw_norm.square())
        block_cap_active.append(
            raw_norm > args.max_local_block_direction_norm
            if args.max_local_block_direction_norm > 0
            else torch.zeros_like(raw_norm, dtype=torch.bool)
        )

    raw_block_norm_squares = torch.stack(raw_block_norm_squares)
    max_raw_block_fraction = raw_block_norm_squares.max() / raw_block_norm_squares.sum().clamp_min(1e-12)
    block_norm_squares = torch.stack(
        [direction_norm(clipped_head).square()]
        + [direction.float().square().sum() for direction in clipped_local]
    )
    max_accepted_block_fraction = block_norm_squares.max() / block_norm_squares.sum().clamp_min(1e-12)
    joint, joint_raw_norm = clip_directions(
        clipped_head + clipped_local, args.max_direction_norm
    )
    head_count = len(clipped_head)
    return (
        joint[:head_count],
        joint[head_count:],
        joint_raw_norm,
        max_raw_block_fraction,
        max_accepted_block_fraction,
        torch.stack(block_cap_active).float().mean(),
    )


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


class RecentObservationReservoir:
    """Fixed-memory ring of recent normalized observations for trust-region guards."""

    def __init__(self, capacity, observation_dim, device):
        if capacity <= 0:
            raise ValueError("kl_anchor_capacity must be positive")
        self.buffer = torch.empty(capacity, observation_dim, device=device)
        self.capacity = capacity
        self.count = 0
        self.cursor = 0

    @torch.no_grad()
    def add(self, observations):
        observations = observations.detach()
        if observations.shape[0] >= self.capacity:
            self.buffer.copy_(observations[-self.capacity :])
            self.count = self.capacity
            self.cursor = 0
            return
        indices = (torch.arange(observations.shape[0], device=observations.device) + self.cursor) % self.capacity
        self.buffer.index_copy_(0, indices, observations)
        self.cursor = (self.cursor + observations.shape[0]) % self.capacity
        self.count = min(self.capacity, self.count + observations.shape[0])

    def observations(self):
        return self.buffer[: self.count]


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
            proposed, _, diagnostics = self.actor_pc.compiled_settle(
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
        else:
            proposed, _, diagnostics = self.actor_pc.settle(
                x, lambda h: self.actor_terminal_grad(h, action_z, args), args, collect_diagnostics
            )
        return self.actor_pc.guard_settlement(
            x,
            proposed,
            lambda h: -args.pc_actor_terminal_coef * self.actor_outputs(h)[2].log_prob(action_z).sum(-1),
            args,
            diagnostics,
        )

    def settle_critic(self, x, args, collect_diagnostics=True):
        if self._compiled_critic_settle is not None:
            proposed, _, diagnostics = self.critic_pc.compiled_settle(
                x,
                self._compiled_critic_settle,
                (self.critic_head.weight,),
                args,
                collect_diagnostics,
            )
        else:
            proposed, _, diagnostics = self.critic_pc.settle(
                x, lambda h: self.critic_terminal_grad(h, args), args, collect_diagnostics
            )
        return self.critic_pc.guard_settlement(
            x,
            proposed,
            lambda h: -args.pc_critic_terminal_coef * self.critic_head(h).squeeze(-1),
            args,
            diagnostics,
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
    def snapshot_critic(self):
        tensors = [self.critic_head.weight, self.critic_head.bias]
        for edge in self.critic_pc.edges:
            tensors.extend([edge.weight, edge.bias])
        return [tensor.clone() for tensor in tensors]

    @torch.no_grad()
    def restore_critic(self, snapshot):
        tensors = [self.critic_head.weight, self.critic_head.bias]
        for edge in self.critic_pc.edges:
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
    if args.pc_compensate_critic_local_score and args.pc_critic_terminal_coef <= 0:
        raise ValueError("critic score compensation requires a positive terminal coefficient")
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
    observation_dim = int(np.prod(envs.single_observation_space.shape))
    kl_anchor = RecentObservationReservoir(args.kl_anchor_capacity, observation_dim, device)

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
            action, action_z, value, deploy_actor, deploy_critic, alpha_raw, beta_raw, _ = agent.act(obs, args)
            kl_anchor.add(obs)
            anchor_obs = kl_anchor.observations()
            anchor_states = agent.deployment(agent.actor_pc, anchor_obs, args)
            _, _, anchor_old_dist = agent.actor_outputs(anchor_states[-1])
            anchor_old_alpha = anchor_old_dist.concentration1.clone()
            anchor_old_beta = anchor_old_dist.concentration0.clone()

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
            actor_head_scores = [alpha_w, alpha_b, beta_w, beta_b]
            critic_head_scores = [value_w, value_b]
            actor_valid = finite_example_rows(
                actor_head_scores + actor_local_scores + list(actor_residuals) + [td_error, actor_delta]
            )
            critic_valid = finite_example_rows(
                critic_head_scores + critic_local_scores + list(critic_residuals) + [td_error, critic_delta]
            )
            actor_head_trace.reset(~actor_valid)
            actor_local_trace.reset(~actor_valid)
            critic_head_trace.reset(~critic_valid)
            critic_local_trace.reset(~critic_valid)
            actor_head_scores = zero_invalid_rows(actor_head_scores, actor_valid)
            actor_local_scores = zero_invalid_rows(actor_local_scores, actor_valid)
            critic_head_scores = zero_invalid_rows(critic_head_scores, critic_valid)
            critic_local_scores = zero_invalid_rows(critic_local_scores, critic_valid)
            critic_local_scores = compensate_critic_nudge_tensors(critic_local_scores, args)
            actor_delta = torch.where(actor_valid, actor_delta, 0.0)
            critic_delta = torch.where(critic_valid, critic_delta, 0.0)

            # Crucial ordering: current eligibility enters before this transition's
            # delta is applied; terminal rows reset only after receiving that delta.
            actor_head_trace.accumulate(actor_head_scores, trace_decay)
            critic_head_trace.accumulate(critic_head_scores, trace_decay)
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
            (
                actor_head_dir,
                actor_local_dir,
                actor_joint_norm,
                actor_max_raw_block_fraction,
                actor_max_accepted_block_fraction,
                actor_block_cap_fraction,
            ) = clip_direction_blocks(actor_head_dir, actor_local_dir, args)
            (
                critic_head_dir,
                critic_local_dir,
                critic_joint_norm,
                critic_max_raw_block_fraction,
                critic_max_accepted_block_fraction,
                critic_block_cap_fraction,
            ) = clip_direction_blocks(critic_head_dir, critic_local_dir, args)

            snapshot = agent.snapshot_actor()
            actor_scale = frac * kl_controller.scale
            apply_head_directions(agent, actor_head_dir, args.actor_learning_rate * actor_scale, actor=True)
            apply_local_directions(agent.actor_pc, actor_local_dir, args.local_learning_rate * actor_scale)
            anchor_new_states = agent.deployment(agent.actor_pc, anchor_obs, args)
            _, _, anchor_new_dist = agent.actor_outputs(anchor_new_states[-1])
            proposed_kl, proposed_anchor_max_kl = beta_kl_stats(
                anchor_old_alpha, anchor_old_beta, anchor_new_dist
            )
            hard_scale = 1.0
            actor_guard_active = False
            if (
                not math.isfinite(proposed_kl)
                or not math.isfinite(proposed_anchor_max_kl)
                or proposed_kl > args.max_kl
                or proposed_anchor_max_kl > args.max_anchor_state_kl
            ):
                # The hierarchy makes KL nonlinear in the joint head/local step.
                # Bisection measures the whole recent-state behavior anchor.
                actor_guard_active = True
                low, high = 0.0, actor_scale
                for _ in range(args.kl_guard_bisection_steps):
                    candidate = 0.5 * (low + high)
                    agent.restore_actor(snapshot)
                    apply_head_directions(agent, actor_head_dir, args.actor_learning_rate * candidate, actor=True)
                    apply_local_directions(agent.actor_pc, actor_local_dir, args.local_learning_rate * candidate)
                    candidate_states = agent.deployment(agent.actor_pc, anchor_obs, args)
                    _, _, candidate_dist = agent.actor_outputs(candidate_states[-1])
                    candidate_kl, candidate_max_kl = beta_kl_stats(
                        anchor_old_alpha, anchor_old_beta, candidate_dist
                    )
                    if (
                        math.isfinite(candidate_kl)
                        and math.isfinite(candidate_max_kl)
                        and candidate_kl <= args.max_kl
                        and candidate_max_kl <= args.max_anchor_state_kl
                    ):
                        low = candidate
                    else:
                        high = candidate
                agent.restore_actor(snapshot)
                hard_scale = low / max(actor_scale, 1e-12)
                actor_scale = low
                apply_head_directions(agent, actor_head_dir, args.actor_learning_rate * actor_scale, actor=True)
                apply_local_directions(agent.actor_pc, actor_local_dir, args.local_learning_rate * actor_scale)
                anchor_new_states = agent.deployment(agent.actor_pc, anchor_obs, args)
                _, _, anchor_new_dist = agent.actor_outputs(anchor_new_states[-1])
            post_kl, post_anchor_max_kl = beta_kl_stats(
                anchor_old_alpha, anchor_old_beta, anchor_new_dist
            )
            if (
                not math.isfinite(post_kl)
                or not math.isfinite(post_anchor_max_kl)
                or post_kl > args.max_kl
                or post_anchor_max_kl > args.max_anchor_state_kl
            ):
                # Exact rollback is the final authority if floating point or local
                # non-monotonicity defeats the search.
                agent.restore_actor(snapshot)
                actor_scale = 0.0
                hard_scale = 0.0
                post_kl = 0.0
                post_anchor_max_kl = 0.0
            kl_controller.adapt(post_kl)
            agent.actor_pc.note_accepted_weight_update(
                actor_local_dir, args.local_learning_rate * actor_scale, args
            )

            critic_snapshot = agent.snapshot_critic()
            critic_scale = frac
            critic_guard_obs = torch.cat([obs, bootstrap_obs], dim=0)
            critic_old_values = torch.cat([value, next_value])
            critic_rms_limit = max(
                args.critic_value_rms_floor,
                args.critic_value_rms_ratio * float(td_error.square().mean().sqrt()),
            )
            apply_head_directions(
                agent, critic_head_dir, args.critic_learning_rate * critic_scale, actor=False
            )
            apply_local_directions(
                agent.critic_pc, critic_local_dir, args.local_learning_rate * critic_scale
            )
            critic_new_values = agent.get_value(critic_guard_obs, args)
            critic_change = critic_new_values - critic_old_values
            critic_proposed_rms = float(critic_change.square().mean().sqrt())
            critic_proposed_max = float(critic_change.abs().max())
            critic_guard_active = (
                not math.isfinite(critic_proposed_rms)
                or not math.isfinite(critic_proposed_max)
                or critic_proposed_rms > critic_rms_limit
                or critic_proposed_max > args.critic_value_max_change
            )
            if critic_guard_active:
                low, high = 0.0, critic_scale
                for _ in range(args.critic_guard_bisection_steps):
                    candidate = 0.5 * (low + high)
                    agent.restore_critic(critic_snapshot)
                    apply_head_directions(
                        agent, critic_head_dir, args.critic_learning_rate * candidate, actor=False
                    )
                    apply_local_directions(
                        agent.critic_pc, critic_local_dir, args.local_learning_rate * candidate
                    )
                    candidate_values = agent.get_value(critic_guard_obs, args)
                    candidate_change = candidate_values - critic_old_values
                    candidate_rms = float(candidate_change.square().mean().sqrt())
                    candidate_max = float(candidate_change.abs().max())
                    if (
                        math.isfinite(candidate_rms)
                        and math.isfinite(candidate_max)
                        and candidate_rms <= critic_rms_limit
                        and candidate_max <= args.critic_value_max_change
                    ):
                        low = candidate
                    else:
                        high = candidate
                agent.restore_critic(critic_snapshot)
                critic_scale = low
                apply_head_directions(
                    agent, critic_head_dir, args.critic_learning_rate * critic_scale, actor=False
                )
                apply_local_directions(
                    agent.critic_pc, critic_local_dir, args.local_learning_rate * critic_scale
                )
                critic_new_values = agent.get_value(critic_guard_obs, args)
                critic_change = critic_new_values - critic_old_values
            critic_post_rms = float(critic_change.square().mean().sqrt())
            critic_post_max = float(critic_change.abs().max())
            if (
                not math.isfinite(critic_post_rms)
                or not math.isfinite(critic_post_max)
                or critic_post_rms > critic_rms_limit
                or critic_post_max > args.critic_value_max_change
            ):
                agent.restore_critic(critic_snapshot)
                critic_scale = 0.0
                critic_post_rms = 0.0
                critic_post_max = 0.0
            agent.critic_pc.note_accepted_weight_update(
                critic_local_dir, args.local_learning_rate * critic_scale, args
            )
            agent.actor_pc.update_precisions(actor_residuals, args, actor_valid)
            critic_precision_residuals = compensate_critic_nudge_tensors(critic_residuals, args)
            agent.critic_pc.update_precisions(critic_precision_residuals, args, critic_valid)
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
            writer.add_scalar("td/actor_nonfinite_fraction", (~actor_valid).float().mean(), global_step)
            writer.add_scalar("td/critic_nonfinite_fraction", (~critic_valid).float().mean(), global_step)
            writer.add_scalar("trust/post_update_kl", post_kl, global_step)
            writer.add_scalar("trust/proposed_kl", proposed_kl, global_step)
            writer.add_scalar("trust/anchored_kl", post_kl, global_step)
            writer.add_scalar("trust/anchored_max_kl", post_anchor_max_kl, global_step)
            writer.add_scalar("trust/proposed_anchor_max_kl", proposed_anchor_max_kl, global_step)
            writer.add_scalar("trust/anchor_size", kl_anchor.count, global_step)
            writer.add_scalar("trust/actor_guard_active", float(actor_guard_active), global_step)
            writer.add_scalar("trust/controller_scale", kl_controller.scale, global_step)
            writer.add_scalar("trust/hard_scale", hard_scale, global_step)
            writer.add_scalar("trust/accepted_actor_scale", actor_scale, global_step)
            writer.add_scalar("trust/critic_proposed_value_rms", critic_proposed_rms, global_step)
            writer.add_scalar("trust/critic_post_value_rms", critic_post_rms, global_step)
            writer.add_scalar("trust/critic_post_value_max", critic_post_max, global_step)
            writer.add_scalar("trust/critic_value_rms_limit", critic_rms_limit, global_step)
            writer.add_scalar("trust/critic_guard_active", float(critic_guard_active), global_step)
            writer.add_scalar("trust/accepted_critic_scale", critic_scale, global_step)
            writer.add_scalar("direction/actor_head", actor_head_norm, global_step)
            writer.add_scalar("direction/critic_head", critic_head_norm, global_step)
            writer.add_scalar("direction/actor_local", actor_local_norm, global_step)
            writer.add_scalar("direction/critic_local", critic_local_norm, global_step)
            writer.add_scalar("direction/actor_joint", actor_joint_norm, global_step)
            writer.add_scalar("direction/critic_joint", critic_joint_norm, global_step)
            writer.add_scalar("direction/actor_max_block_fraction", actor_max_raw_block_fraction, global_step)
            writer.add_scalar("direction/critic_max_block_fraction", critic_max_raw_block_fraction, global_step)
            writer.add_scalar(
                "direction/actor_max_accepted_block_fraction",
                actor_max_accepted_block_fraction,
                global_step,
            )
            writer.add_scalar(
                "direction/critic_max_accepted_block_fraction",
                critic_max_accepted_block_fraction,
                global_step,
            )
            writer.add_scalar("direction/actor_block_cap_fraction", actor_block_cap_fraction, global_step)
            writer.add_scalar("direction/critic_block_cap_fraction", critic_block_cap_fraction, global_step)
            writer.add_scalar("pc/actor_steps", actor_diag["steps"], global_step)
            writer.add_scalar("pc/critic_steps", critic_diag["steps"], global_step)
            writer.add_scalar("pc/actor_inference_step_mean", actor_diag["mean_step"], global_step)
            writer.add_scalar("pc/critic_inference_step_mean", critic_diag["mean_step"], global_step)
            writer.add_scalar("pc/actor_max_curvature", actor_diag["max_curvature"], global_step)
            writer.add_scalar("pc/critic_max_curvature", critic_diag["max_curvature"], global_step)
            writer.add_scalar("pc/actor_max_condition", actor_diag["max_condition"], global_step)
            writer.add_scalar("pc/critic_max_condition", critic_diag["max_condition"], global_step)
            writer.add_scalar("pc/actor_mean_condition", actor_diag["mean_condition"], global_step)
            writer.add_scalar("pc/critic_mean_condition", critic_diag["mean_condition"], global_step)
            writer.add_scalar("pc/actor_energy_delta", actor_diag["energy_delta"], global_step)
            writer.add_scalar("pc/critic_energy_delta", critic_diag["energy_delta"], global_step)
            writer.add_scalar(
                "pc/actor_backtrack_scale_mean", actor_diag["backtrack_scale_mean"], global_step
            )
            writer.add_scalar(
                "pc/critic_backtrack_scale_mean", critic_diag["backtrack_scale_mean"], global_step
            )
            writer.add_scalar(
                "pc/actor_energy_reject_fraction", actor_diag["energy_reject_fraction"], global_step
            )
            writer.add_scalar(
                "pc/critic_energy_reject_fraction", critic_diag["energy_reject_fraction"], global_step
            )
            writer.add_scalar("pc/actor_state_clip_fraction", actor_diag["state_clip_fraction"], global_step)
            writer.add_scalar("pc/critic_state_clip_fraction", critic_diag["state_clip_fraction"], global_step)
            writer.add_scalar(
                "pc/actor_curvature_relative_drift",
                agent.actor_pc._curvature_accumulated_relative_drift,
                global_step,
            )
            writer.add_scalar(
                "pc/critic_curvature_relative_drift",
                agent.critic_pc._curvature_accumulated_relative_drift,
                global_step,
            )
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
                    * args.local_learning_rate * critic_scale,
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
