"""Sphere PPO with passive future-utility plasticity.

The gate does not try to judge the current gradient's magnitude or sign.  It
predicts whether the candidate Adam displacement for each incoming row will be
useful on the next rollout.  The next rollout is already required by PPO, so
its gradient supplies a delayed, passive target:

    utility_t = -cos(candidate_update_t, gradient_{t+1})

A small shared predictor maps causal row features to the expected normalized
utility and its uncertainty.  Ambiguous predictions leave the baseline step
unchanged; confident positive/negative predictions amplify/suppress the
realized post-Adam row update.  The predictor is trained only from delayed
future gradients and never changes the current PPO loss.

This is a new method, not a v10 throughput variant.  The legacy residual-energy
GLS and instantaneous SNR paths are disabled by default because they do not
predict future usefulness.
"""
from __future__ import annotations

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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import ppo_continuous_action_sphere_sdplast_v10 as base
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
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


@dataclass
class Args(base.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    gls_weights: bool = False
    """legacy instantaneous residual-energy weighting; off for this method"""
    snr_level: bool = False
    """legacy instantaneous batch-SNR level; off for this method"""
    utility_lr: float = 1e-3
    """learning rate for the delayed utility forecaster"""
    utility_beta: float = 0.97
    """EMA decay for the delayed utility moments"""
    utility_warmup: int = 32
    """number of delayed labels before a row can be gated"""
    utility_floor: float = 0.25
    """minimum post-Adam multiplier at confident negative utility"""
    utility_ceiling: float = 2.0
    """maximum post-Adam multiplier at confident positive utility"""
    utility_temperature: float = 0.10
    """uncertainty floor in normalized utility units"""
    utility_gain: float = 1.0
    """sensitivity of the bounded future-utility multiplier"""
    utility_clip: float = 1.0
    """gradient-norm clip for the utility forecaster"""


def validate_args(args):
    args = base.validate_args(args)
    if not (0.0 < args.utility_lr) or not (0.0 < args.utility_beta < 1.0):
        raise ValueError("utility_lr must be positive and utility_beta must lie in (0, 1)")
    if args.utility_warmup < 1 or args.utility_floor <= 0.0:
        raise ValueError("utility_warmup must be positive and utility_floor must be positive")
    if args.utility_ceiling < 1.0 or args.utility_floor >= args.utility_ceiling:
        raise ValueError("utility ceiling must be at least one and exceed the floor")
    if args.utility_temperature <= 0.0 or args.utility_gain <= 0.0:
        raise ValueError("utility temperature and gain must be positive")
    return args


class UtilityForecaster(nn.Module):
    """Shared causal row forecaster for mean utility and log variance."""

    def __init__(self, features=6):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(features, 32), nn.SiLU(), nn.Linear(32, 2))
        # Zero output at initialization means the learner is exactly ungated
        # until delayed evidence is available.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features):
        output = self.body(features)
        return output[:, 0].clamp(-2.0, 2.0), output[:, 1].clamp(-5.0, 2.0)


class ProspectiveUtilityStepper:
    """Delayed future-utility predictor and post-Adam row gate.

    Every site is represented by one row vector including its bias when present.
    The gate is deliberately row-level: it predicts whether the complete
    incoming direction is worth applying, avoiding a false claim that a scalar
    row multiplier can discover an input-space direction.
    """

    def __init__(self, sites, args, device):
        self.layers = [module for _, module in sites]
        self.args = args
        self.device = device
        self.forecaster = UtilityForecaster().to(device)
        self.forecaster_optimizer = optim.Adam(
            self.forecaster.parameters(), lr=args.utility_lr, eps=1e-8, fused=True
        )
        self.beta = args.utility_beta
        self.rows = [layer.weight.shape[0] for layer in self.layers]
        self.grad_ema = [torch.zeros(n, device=device) for n in self.rows]
        self.previous_grad = [torch.zeros_like(layer.weight) for layer in self.layers]
        self.previous_update = [torch.zeros_like(layer.weight) for layer in self.layers]
        self.previous_bias_grad = [
            torch.zeros(n, device=device) if layer.bias is not None else None
            for layer, n in zip(self.layers, self.rows)
        ]
        self.previous_bias_update = [
            torch.zeros(n, device=device) if layer.bias is not None else None
            for layer, n in zip(self.layers, self.rows)
        ]
        self.utility_mean = [torch.zeros(n, device=device) for n in self.rows]
        self.utility_square = [torch.zeros(n, device=device) for n in self.rows]
        self.utility_count = [torch.zeros(n, device=device) for n in self.rows]
        self.snapshots = [torch.empty_like(layer.weight) for layer in self.layers]
        self.bias_snapshots = [
            torch.empty_like(layer.bias) if layer.bias is not None else None
            for layer in self.layers
        ]
        self.gates = [torch.ones(n, device=device) for n in self.rows]
        self.current_features = None
        self.last_forecast_loss = torch.zeros((), device=device)
        self.last_label_mean = torch.zeros((), device=device)
        self.last_gate_mean = torch.ones((), device=device)
        self.last_gate_std = torch.zeros((), device=device)
        self.last_positive_fraction = torch.zeros((), device=device)
        self.updates = 0

    @staticmethod
    def _row_vector(layer, weight_value, bias_value):
        if bias_value is None:
            return weight_value
        return torch.cat((weight_value, bias_value.unsqueeze(1)), dim=1)

    @staticmethod
    def _row_norm(value):
        return value.square().mean(1).add(1e-12).sqrt()

    def _current_gradients(self):
        vectors = []
        for layer in self.layers:
            weight_grad = layer.weight.grad
            if weight_grad is None:
                weight_grad = torch.zeros_like(layer.weight)
            bias_grad = None if layer.bias is None else layer.bias.grad
            if bias_grad is None and layer.bias is not None:
                bias_grad = torch.zeros_like(layer.bias)
            vectors.append(self._row_vector(layer, weight_grad, bias_grad))
        return vectors

    def _features_and_labels(self, gradients):
        features, labels = [], []
        has_previous = self.current_features is not None
        for index, (layer, gradient) in enumerate(zip(self.layers, gradients)):
            scale = self.grad_ema[index]
            norm = self._row_norm(gradient)
            scale.mul_(self.beta).add_(norm, alpha=1.0 - self.beta)
            log_ratio = (norm / (scale + 1e-12)).clamp(1e-4, 1e4).log()
            previous = self._row_vector(
                layer,
                self.previous_grad[index],
                self.previous_bias_grad[index],
            )
            previous_norm = self._row_norm(previous)
            cosine = (gradient * previous).sum(1) / (norm * previous_norm + 1e-12)
            mean = self.utility_mean[index]
            variance = (self.utility_square[index] - mean.square()).clamp_min(0.0)
            count = self.utility_count[index]
            row_features = torch.stack(
                (
                    log_ratio,
                    cosine,
                    (norm + 1e-12).log(),
                    mean,
                    variance.sqrt(),
                    (count / float(self.args.utility_warmup)).clamp(0.0, 1.0),
                ),
                dim=1,
            )
            features.append(row_features)
            if has_previous:
                previous_update = self._row_vector(
                    layer,
                    self.previous_update[index],
                    self.previous_bias_update[index],
                )
                previous_update_norm = self._row_norm(previous_update)
                label = -(previous_update * gradient).sum(1) / (
                    previous_update_norm * norm + 1e-12
                )
                labels.append(label.clamp(-1.0, 1.0))
            self.previous_grad[index].copy_(gradient[:, :-1] if layer.bias is not None else gradient)
            if layer.bias is not None:
                self.previous_bias_grad[index].copy_(gradient[:, -1])
        return features, labels if has_previous else None

    def _learn_delayed_target(self, labels):
        if labels is None:
            self.last_forecast_loss.zero_()
            self.last_label_mean.zero_()
            return
        old_features = torch.cat(self.current_features, dim=0).detach()
        target = torch.cat(labels, dim=0).detach()
        predicted_mean, predicted_logvar = self.forecaster(old_features)
        residual = target - predicted_mean
        loss = 0.5 * (residual.square() * (-predicted_logvar).exp() + predicted_logvar).mean()
        self.forecaster_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.forecaster.parameters(), self.args.utility_clip)
        self.forecaster_optimizer.step()
        self.last_forecast_loss.copy_(loss.detach())
        self.last_label_mean.copy_(target.mean())

        offset = 0
        for index, row_labels in enumerate(labels):
            count = self.utility_count[index]
            mean = self.utility_mean[index]
            square = self.utility_square[index]
            count.add_(1.0)
            mean.mul_(self.beta).add_(row_labels, alpha=1.0 - self.beta)
            square.mul_(self.beta).addcmul_(row_labels, row_labels, value=1.0 - self.beta)
            offset += row_labels.numel()

    def observe(self):
        """Consume current gradients and train on the previous update's label."""
        gradients = self._current_gradients()
        features, labels = self._features_and_labels(gradients)
        self._learn_delayed_target(labels)
        flat_features = torch.cat(features, dim=0)
        mean, logvar = self.forecaster(flat_features)
        std = F.softplus(logvar) + self.args.utility_temperature
        confidence = torch.tanh(mean / std)
        raw_gate = torch.exp(self.args.utility_gain * confidence)
        raw_gate = raw_gate.clamp(self.args.utility_floor, self.args.utility_ceiling)
        gates = []
        offset = 0
        for count, width in zip(self.utility_count, self.rows):
            predicted = raw_gate[offset : offset + width]
            ready = (count >= float(self.args.utility_warmup)).to(predicted.dtype)
            gates.append(ready * predicted + (1.0 - ready))
            offset += width
        self.gates = gates
        self.current_features = [value.detach() for value in features]
        self.last_gate_mean.copy_(torch.cat(gates).mean())
        self.last_gate_std.copy_(torch.cat(gates).std())
        self.last_positive_fraction.copy_((torch.cat(gates) > 1.0).to(torch.float32).mean())
        return gradients

    @torch.no_grad()
    def stash(self):
        for index, layer in enumerate(self.layers):
            self.snapshots[index].copy_(layer.weight)
            if layer.bias is not None:
                self.bias_snapshots[index].copy_(layer.bias)

    @torch.no_grad()
    def apply_and_remember(self):
        """Scale the realized optimizer displacement and save it as the next label."""
        for index, layer in enumerate(self.layers):
            gate = self.gates[index]
            weight_update = layer.weight - self.snapshots[index]
            weight_update.mul_(gate.unsqueeze(1))
            layer.weight.copy_(self.snapshots[index] + weight_update)
            self.previous_update[index].copy_(weight_update)
            if layer.bias is not None:
                bias_update = layer.bias - self.bias_snapshots[index]
                bias_update.mul_(gate)
                layer.bias.copy_(self.bias_snapshots[index] + bias_update)
                self.previous_bias_update[index].copy_(bias_update)
        self.updates += 1

    def metrics(self):
        return {
            "plasticity/utility_nll": self.last_forecast_loss,
            "plasticity/future_utility": self.last_label_mean,
            "plasticity/gate_mean": self.last_gate_mean,
            "plasticity/gate_std": self.last_gate_std,
            "plasticity/positive_gate_fraction": self.last_positive_fraction,
        }




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
    configure_runtime(
        cudnn_deterministic=args.torch_deterministic,
        matmul_precision="highest",
        allow_tf32=False,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text(
            "policy",
            "Beta host actor; hypersphered SiTU-GLU trunk; passive delayed future-utility row gating; "
            "legacy residual-energy and instantaneous SNR gates disabled",
        )
        writer.add_text(
            "utility_contract",
            "utility=-cos(previous_realized_row_update,current_raw_row_gradient); "
            "gate applies after Adam and leaves optimizer moments untouched",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = base.Agent(envs, args).to(device)
        net_params = agent.network_parameters()
        optimizer = optim.Adam(
            net_params,
            lr=args.learning_rate,
            eps=1e-5,
            fused=True,
        )
        plasticity = ProspectiveUtilityStepper(agent.plastic_sites, args, device)

        def value_model(observations):
            return agent.get_value(observations)

        def rollout_statistics(observations, native):
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return base.ppo_loss(
                agent,
                observations,
                native,
                old_logprobs,
                advantages,
                returns,
                old_values,
                args,
            )

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(
                value_model,
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False},
            )
            loss_model = torch.compile(
                loss_model,
                mode=args.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (
            buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high)
        )
        beta_head = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = beta_head(host_actor(observations), sampler)
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
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
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
                optimizer.param_groups[0]["lr"] = (
                    1.0 - (iteration - 1.0) / args.num_iterations
                ) * args.learning_rate
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
                    next_obs_np, transition_obs = obs_norm.normalize_step(
                        raw_obs, terms, truncs, infos
                    )
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(
                        step,
                        reward,
                        terms,
                        truncs,
                        observations=obs_step,
                        native_actions=native,
                    )
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs = rollout_statistics(b_obs, b_native)
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

            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(
                        args.batch_size,
                        args.minibatch_size,
                        device,
                        shuffle_generator,
                    ):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            b_returns[indices],
                            b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        plasticity.clear_probes()
                        loss.backward()
                        plasticity.observe()
                        plasticity.stash()
                        nn.utils.clip_grad_norm_(net_params, args.max_grad_norm)
                        optimizer.step()
                        plasticity.apply_and_remember()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = gather_metrics(
                {
                    "losses/policy_loss": last[0],
                    "losses/value_loss": last[1],
                    "losses/entropy": last[2],
                    "losses/old_approx_kl": last[3],
                    "losses/approx_kl": last[4],
                    "losses/clipfrac": update_metrics[:updates, 5].mean(),
                    "losses/explained_variance": explained_variance(b_values, b_returns),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    **plasticity.metrics(),
                }
            )
            if any(
                not np.isfinite(value)
                for name, value in logged.items()
                if name != "losses/explained_variance"
            ):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar(
                "charts/interval_SPS",
                (global_step - interval_step) / (now - interval_start),
                global_step,
            )
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            interval_start, interval_step = time.perf_counter(), global_step

    finally:
        resources.close()


if __name__ == "__main__":
    main()
