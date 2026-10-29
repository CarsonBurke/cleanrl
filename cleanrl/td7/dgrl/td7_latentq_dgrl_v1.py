# TD7 LATENT-Q DGRL v1 — Distance-Based Updates and Sampled Dynamic Neighborhoods (Hoppe et al.
# 2026, arXiv 2602.08616) on delight reference v10's critic, replay and cadence.
# DBU: at replay states, M antithetic draws u = mu + sigma*eps (plus the mode) are scored by the
# frozen point head at their clipped actions; mu regresses (L2) onto the softmax(Q / (tau * batch
# RMS))-weighted mean of the draws. To first order the target step is sigma^2 * grad Q / (tau * rms): a
# Fisher-preconditioned, scale-free PG step whose target never leaves the sampled cloud. Beyond
# first order the tilt sees curvature and multiple modes that a score update averages away.
# Sigma takes a Gaussian-NLL step toward the reliability-corrected weighted variance of the draws,
# the matching CEM/CMA second moment: unchanged on a linear critic, wider along convex and
# narrower along concave directions. The v8 entropy floor stays; nothing differentiates Q.
# SDN (sdn_candidates > 0): each env step draws K Gaussian candidates, scores them with the
# point head on the GPU and executes one by rank softmax tau_e^rank. Replay stores the executed
# draw; the Hermite centre lane then sees tilted eps, a second-order bias on a smooth critic.
# Hypothesis: a critic-tilted regression target is a better improvement operator than a gated
# score on the same draws, and Q-guided selection raises the quality of what is executed.
import os
import math
import random
import signal
import time
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.graph_step import capture_steps
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl_utils.checkpointing import CheckpointCadence, atomic_save


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    compile: bool = True
    """compile the losses with Inductor before capturing the train-step graphs"""
    resume: str = ""
    """resume a DGRL v1 rollout-boundary checkpoint; MuJoCo trajectories restart"""
    checkpoint_path: str = ""
    """checkpoint destination; default runs/<run_name>/checkpoint.pt"""
    checkpoint_interval_seconds: float = 300.0
    """atomic checkpoints at completed rollouts, every 300-600 seconds and on exit"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_steps: int = 1024
    """steps per environment per rollout"""
    env_threads: int = 2
    """maximum physics threads"""
    actor_sale: bool = True
    """use the referenced run's fixed SALE state embedding in the actor"""
    dbu_samples: int = 32
    """even number M of antithetic Gaussian draws per replay state for the DBU target"""
    dbu_temperature: float = 1.0
    """softmax temperature in units of the batch RMS of Q(draw) - Q(mode); small -> argmax"""
    dbu_include_mode: bool = True
    """add the mode clip(mu) as a candidate, so a flat or concave critic pulls the target home"""
    sdn_candidates: int = 0
    """K Gaussian candidates per env step scored by the point head (0: plain Gaussian acting)"""
    sdn_rank_temperature: float = 0.8
    """SDN selection probability proportional to tau_e^rank (rank 0 = highest Q)"""
    max_grad_norm: float = 0.5
    """clip encoder, critic and actor gradients; absent from original preaction v5"""
    updates_per_step: float = 1.0
    """replay updates per environment step"""
    warmup_rollouts: int = 1
    """recorded rollouts of latent-free exploration (mu=0, sigma=warmup_sigma) before training starts"""
    warmup_sigma: float = 0.6
    """std of the Gaussian warmup policy (the TD7 uniform warmup has no Gaussian latent; 0.6 ~ its std)"""
    hermite_order: int = 2
    """0: plain Q~ regression; 1: + sigma*eps*dQ~/dmu; 2: + (eps^2-1)*sigma/2*dQ~/dsigma"""
    eval_interval: int = 2
    """evaluate the policy mean every N rollouts"""
    eval_eps: int = 10
    """number of evaluation episodes (run as one batched rollout)"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update_rate: int = 250
    """hard-update the target networks and fixed encoders every N training steps"""
    entropy_floor: bool = True
    """retain entropy bonus and alpha tuner; false leaves only the weighted-variance regression"""
    target_sigma: float = 0.1
    """minimum mean log-std entropy target for the positive alpha tuner; not an upper sigma bound"""
    target_smoothing_sigma: float = 0.2
    """std passed to the learned distribution-center target query; inherited TD7 value 0.2"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer (replay mode)"""
    policy_freq: int = 2
    """the frequency of training the policy (replay mode)"""
    lap_alpha: float = 0.4
    """LAP prioritization exponent"""
    min_priority: float = 1.0
    """LAP minimum priority (and Huber loss threshold)"""
    zs_dim: int = 256
    """dimensionality of the SALE embeddings"""
    hidden_dim: int = 256
    """hidden layer width of all networks"""
    encoder_lr: float = 3e-4
    """the learning rate of the encoder optimizer"""
    critic_lr: float = 3e-4
    """the learning rate of the critic optimizer"""
    actor_lr: float = 3e-4
    """the learning rate of the actor optimizer"""


# Log-std bounds of the state-dependent Gaussian: sigma in [~6.7e-3, 1] for actions in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def avg_l1_norm_jvp(x, t, eps=1e-8):
    """avg_l1_norm and its forward-mode derivative along tangent t."""
    scale = x.abs().mean(-1, keepdim=True).clamp(min=eps)
    y = x / scale
    return y, (t - y * (x.sign() * t).mean(-1, keepdim=True)) / scale


def elu_jvp(x, t):
    return F.elu(x), torch.where(x > 0, t, t * x.clamp(max=0).exp())


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.sale = zs_dim > 0  # zs_dim=0: state-only actor, ``zs`` is ignored
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.l3_logstd = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        if self.sale:
            a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        log_std = torch.tanh(self.l3_logstd(a))
        return torch.tanh(self.l3(a)), LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs_dim = zs_dim
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        return avg_l1_norm(self.zs3(zs))

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        return self.zsa3(zsa)

    def zsa_jvp(self, zs, action, t_action):
        """zsa(zs, action) and its derivative along an action tangent (zs held fixed)."""
        h, t = elu_jvp(self.zsa1(torch.cat([zs, action], 1)), t_action @ self.zsa1.weight[:, self.zs_dim :].T)
        h, t = elu_jvp(self.zsa2(h), t @ self.zsa2.weight.T)
        return self.zsa3(h), t @ self.zsa3.weight.T


class EnsembleLinear(nn.Module):
    """``members`` independent nn.Linear layers applied to (members, batch, in) as one baddbmm."""

    def __init__(self, members, in_features, out_features):
        super().__init__()
        bound = in_features**-0.5  # nn.Linear's default init bound, for weight and bias
        self.weight = nn.Parameter(torch.empty(members, in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, out_features).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)

    def jvp(self, x, t):
        """Value and tangent in one bmm: the tangent rows get the weight but not the bias."""
        out = torch.bmm(torch.cat([x, t], 1), self.weight)
        return out[:, : x.shape[1]] + self.bias, out[:, x.shape[1] :]


class PreActionLatentTwinCritic(nn.Module):
    """Fused twin critic conditioned on (policy mean, policy std, pre-action latent)."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim, self.state_dim = hdim, state_dim
        latent_dim = 3 * action_dim  # mean | std | pre-action latent
        self.q0 = nn.Linear(state_dim + latent_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        bound = (state_dim + latent_dim) ** -0.5  # per-member nn.Linear init
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(self, state, latent, pre_action, zsa, zs, t_latent=None, t_pre_action=None, t_zsa=None):
        batch = state.shape[0]
        latent_input = torch.cat([latent, pre_action], 1)
        pre = self.q0(torch.cat([state, latent_input], 1)).view(batch, 2, self.hdim)
        embeddings = torch.cat([zsa, zs], 1).expand(2, batch, -1)
        if t_latent is None:
            h = avg_l1_norm(pre).transpose(0, 1)
            h = F.elu(self.q1(torch.cat([h, embeddings], 2)))
            h = F.elu(self.q2(h))
            return self.q3(h).squeeze(-1).transpose(0, 1)
        t_latent_input = torch.cat([t_latent, t_pre_action], 1)
        t_pre = (t_latent_input @ self.q0.weight[:, self.state_dim :].T).view(batch, 2, self.hdim)
        h, t = avg_l1_norm_jvp(pre, t_pre)
        t_embeddings = torch.cat([t_zsa, torch.zeros_like(zs)], 1).expand(2, batch, -1)
        h = torch.cat([h.transpose(0, 1), embeddings], 2)
        t = torch.cat([t.transpose(0, 1), t_embeddings], 2)
        h, t = elu_jvp(*self.q1.jvp(h, t))
        h, t = elu_jvp(*self.q2.jvp(h, t))
        q, t = self.q3.jvp(h, t)
        return q.squeeze(-1).transpose(0, 1), t.squeeze(-1).transpose(0, 1)


class HostPolicy:
    """NumPy mirror of the actor (and fixed_encoder.zs when it feeds the actor) for acting on the host.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst).
    """

    def __init__(self, encoder: Encoder, actor: Actor):
        self.sale = actor.sale
        encoder_sources = [
            (encoder.zs1.weight,), (encoder.zs1.bias,), (encoder.zs2.weight,), (encoder.zs2.bias,),
            (encoder.zs3.weight,), (encoder.zs3.bias,),
        ]
        self._sources = (encoder_sources if self.sale else []) + [
            (actor.l0.weight,), (actor.l0.bias,),
            (actor.l1.weight,), (actor.l1.bias,), (actor.l2.weight,), (actor.l2.bias,),
            (actor.l3.weight, actor.l3_logstd.weight), (actor.l3.bias, actor.l3_logstd.bias),
        ]
        shapes = [(sum(t.shape[0] for t in group),) + tuple(group[0].shape[1:]) for group in self._sources]
        self._flat = np.zeros(sum(int(np.prod(s)) for s in shapes), dtype=np.float32)
        self._views, offset = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            view = self._flat[offset : offset + size].reshape(shape)
            self._views.append(view.T if view.ndim == 2 else view)  # weights as (in, out)
            offset += size
        self.action_dim = actor.l3.out_features

    def refresh(self):
        flat = torch.cat([t.detach().reshape(-1) for group in self._sources for t in group])
        self._flat[:] = flat.cpu().numpy()

    def __call__(self, state):
        w0, b0, wl1, bl1, wl2, bl2, wh, bh = self._views[-8:]
        a = state @ w0 + b0
        a /= np.maximum(np.abs(a).mean(-1, keepdims=True), 1e-8)
        if self.sale:
            w1, b1, w2, b2, w3, b3 = self._views[:6]
            zs = state @ w1 + b1
            zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
            zs = zs @ w2 + b2
            zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
            zs = zs @ w3 + b3
            zs /= np.maximum(np.abs(zs).mean(-1, keepdims=True), 1e-8)
            a = np.concatenate([a, zs], 1)
        a = np.maximum(a @ wl1 + bl1, 0)
        a = np.maximum(a @ wl2 + bl2, 0)
        out = a @ wh + bh
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(out[:, self.action_dim :]) + 1)
        return np.tanh(out[:, : self.action_dim]), log_std


TARGET_BOUND = 0.999  # tanh(3.8): a pull to exactly +-1 would chase an unbounded pre-activation

DBU_DIAGNOSTICS = (
    "q_spread_rms", "effective_sample_fraction", "mode_weight", "target_step_sigma",
    "sigma_target_ratio", "clipped_action_fraction",
)


def dbu_targets(candidates, q, draw_count, temperature, include_mode):
    """DBU's softmax-average mean target and the matching weighted variance.

    candidates: (C, batch, action) unclipped pre-actions whose clipped actions the critic
    scored as q: (C, batch); the first draw_count rows are the Gaussian draws and, with
    include_mode, the last row is mu. Logits are within-state Q deviations in units of their
    batch RMS, so the temperature is dimensionless and states where the action barely matters
    keep near-uniform weights (an antithetic cloud then averages back to the mode).
    The mean target averages unclipped candidates: averaging clipped ones would pull a
    symmetric cloud inside a saturated mu even when Q rises into the bound. It is clamped to
    the range tanh reaches. The variance uses the draws only, about their own weighted mean,
    with the reliability correction 1 / (1 - sum w^2): unbiased for draw-independent weights.
    """
    deviation = q - q.mean(0)
    rms = deviation.square().mean().sqrt()
    logits = deviation / (rms.clamp(min=1e-8) * temperature)
    weight = torch.softmax(logits, 0)
    target = (weight[..., None] * candidates).sum(0).clamp(-TARGET_BOUND, TARGET_BOUND)
    draws = candidates[:draw_count]
    draw_weight = torch.softmax(logits[:draw_count], 0)
    center = (draw_weight[..., None] * draws).sum(0)
    spread = (draw_weight[..., None] * (draws - center).square()).sum(0)
    variance = spread / (1 - draw_weight.square().sum(0))[..., None].clamp(min=1e-6)
    effective = 1 / weight.square().sum(0)
    mode_weight = weight[-1].mean() if include_mode else torch.zeros_like(rms)
    return target, variance, torch.stack([rms, (effective / len(q)).mean(), mode_weight])


def dbu_losses(mu, log_std, target, variance):
    """L2 regression of mu onto the target, and the log-sigma half of a Gaussian NLL.

    d/dlog_std of the scale term is (1 - variance / sigma^2) / 2: zero at an unbiased
    variance estimate, unlike log-space regression, whose Jensen gap shrinks sigma.
    """
    location = 0.5 * (mu - target.detach()).square().sum(-1).mean()
    scale = (0.5 * log_std + 0.25 * variance.detach() * (-2 * log_std).exp()).sum(-1).mean()
    return location, scale


def rank_selection_probabilities(count, temperature):
    """SDN's tau_e^rank selection distribution over candidates sorted by descending Q."""
    probability = temperature ** torch.arange(count, dtype=torch.float64)
    return (probability / probability.sum()).float()


def accumulate_clipping_statistics(statistics, norm, max_grad_norm):
    """Accumulate norm sum/count, clip count, multiplier sum and max on device."""
    with torch.no_grad():
        norm = norm.detach()
        multiplier = (max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
        statistics.add_(torch.stack([
            norm, torch.ones_like(norm), (multiplier < 1).to(norm.dtype),
            multiplier, torch.zeros_like(norm),
        ]))
        torch.maximum(statistics[4], norm, out=statistics[4])


def effective_entropy_alpha(log_alpha, entropy_floor):
    """The disabled floor has exactly zero actor entropy coefficient."""
    return log_alpha.detach().exp() if entropy_floor else torch.zeros_like(log_alpha)


def gaussian_kl(mu_old, sigma_old, mu, log_std):
    """Per-state KL(old || new) of diagonal Gaussians, split into its mean and scale terms."""
    mean_term = (0.5 * ((mu_old - mu) / log_std.exp()).square()).sum(-1)
    ratio = sigma_old.log() - log_std
    scale_term = (0.5 * (2 * ratio).exp() - ratio - 0.5).sum(-1)
    return mean_term, scale_term


CHECKPOINT_MODELS = ("actor", "actor_target", "critic", "critic_target", "encoder",
                     "fixed_encoder", "fixed_encoder_target")
CHECKPOINT_OPTIMIZERS = ("encoder_optimizer", "critic_optimizer", "actor_optimizer",
                         "alpha_optimizer")
CHECKPOINT_BUFFERS = ("last", "max_priority", "running_max", "running_min",
                      "max_target", "min_target", "metrics", "fresh", "clipping_statistics", "sdn_statistics")
CHECKPOINT_RUNTIME_ARGS = {"exp_name", "total_timesteps", "compile", "env_threads",
                           "eval_interval", "eval_eps", "resume", "checkpoint_path",
                           "checkpoint_interval_seconds"}


def checkpoint_to_cpu(value):
    """Copy mutable tensors; the file snapshot must not alias live training state."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: checkpoint_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [checkpoint_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_to_cpu(item) for item in value)
    return value


def restore_optimizer_in_place(optimizer, saved):
    """Restore Adam without replacing any tensors referenced by CUDA graphs."""
    if len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer group count differs")
    with torch.no_grad():
        for group, previous in zip(optimizer.param_groups, saved["param_groups"]):
            if len(group["params"]) != len(previous["params"]):
                raise ValueError("checkpoint optimizer parameter count differs")
            for key, value in previous.items():
                if key != "params" and group.get(key) != value:
                    raise ValueError(f"checkpoint optimizer setting differs: {key}")
            for parameter, index in zip(group["params"], previous["params"]):
                # A disabled alpha optimizer has no Adam state. Do not create an
                # empty defaultdict entry while restoring its exact checkpoint.
                state, prior_state = optimizer.state.get(parameter, {}), saved["state"].get(index, {})
                if state.keys() != prior_state.keys():
                    raise ValueError("checkpoint optimizer state differs from captured state")
                for key, value in prior_state.items():
                    current = state[key]
                    if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
                        if current.shape != value.shape or current.dtype != value.dtype:
                            raise ValueError(f"checkpoint optimizer tensor differs: {key}")
                        current.copy_(value)
                    elif current != value:
                        raise ValueError(f"checkpoint non-tensor optimizer state differs: {key}")


def validate_checkpoint_args(checkpoint, args):
    if checkpoint.get("format") != "td7_latentq_dgrl_v1" or checkpoint.get("version") != 1:
        raise ValueError("expected a DGRL v1 training checkpoint")
    previous = checkpoint["args"]
    current = vars(args)
    for key in previous.keys() | current.keys():
        if key not in CHECKPOINT_RUNTIME_ARGS and previous.get(key) != current.get(key):
            raise ValueError(f"resume algorithm argument differs: {key}")


def capture_rng_state(rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "host": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng_state(state, rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    rng.bit_generator.state = state["host"]
    torch.set_rng_state(state["torch"])
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA device count differs")
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, agent, rng, global_step, completed_rollouts):
    checkpoint = {"format": "td7_latentq_dgrl_v1", "version": 1,
                  "args": vars(agent.args).copy(), "global_step": global_step,
                  "completed_rollouts": completed_rollouts,
                  "agent": agent.checkpoint_state(), "rng": capture_rng_state(rng),
                  "environment_resume": "fresh stochastic phase warmup; MuJoCo state not saved"}
    atomic_save(path, lambda staging: torch.save(checkpoint, staging))
    print(f"checkpoint_saved={path}, global_step={global_step}, "
          f"training_steps={agent.training_steps}", flush=True)


class TD7Agent:
    def __init__(self, state_dim, action_dim, args: Args, device, writer: SummaryWriter):
        assert args.hermite_order in (0, 1, 2), "hermite_order must be 0, 1 or 2"
        if args.dbu_samples < 2 or args.dbu_samples % 2:
            raise ValueError("dbu_samples must be a positive even number >= 2")
        if not np.isfinite(args.dbu_temperature) or args.dbu_temperature <= 0:
            raise ValueError("dbu_temperature must be finite and positive")
        if args.sdn_candidates < 0 or args.sdn_candidates == 1:
            raise ValueError("sdn_candidates must be 0 (off) or at least 2")
        if not 0 < args.sdn_rank_temperature <= 1:
            raise ValueError("sdn_rank_temperature must be in (0, 1]")
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim if args.actor_sale else 0
        self.actor = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.critic = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        targets = [
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ]
        targets.append((self.actor_target, self.actor))
        for target, source in targets:
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        self._hard_updates = [
            (list(self.critic_target.parameters()), list(self.critic.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]
        optimizers = [self.encoder_optimizer, self.critic_optimizer]

        # The entropy budget and alpha autotuner are exactly v8's replay formulation.
        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self._hard_updates.insert(0, (list(self.actor_target.parameters()), list(self.actor.parameters())))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        # DBU's scale gradient is dimensionless and at most 1/2 per dimension, so alpha = 1 (v10's
        # start, sized for raw-Q scores) would pin sigma at its ceiling until the tuner caught up.
        self.log_alpha = torch.full((), math.log(0.1), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers += [self.actor_optimizer, self.alpha_optimizer]

        # Replay rows retain both the clipped environment action and its pre-action latent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a), ("next_state", s), ("reward", 1), ("not_done", 1)):
            self._fields[name] = slice(offset, offset + width)
            offset += width
        self.row_dim = offset
        self.storage = torch.zeros(args.buffer_size, self.row_dim, device=device)
        self.priority = torch.zeros(args.buffer_size, device=device)  # zero == empty slot, never sampled
        # Index of the last filled slot. A parallel scan can drift a few ulps through the all-zero
        # tail in either direction, so draws are scaled by csum[last] and indices clamped to last.
        self.last = torch.zeros(1, dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0
        self.rollout_size = args.num_envs * args.num_steps
        self.updates_per_rollout = round(self.rollout_size * args.updates_per_step)
        self.staging = torch.zeros(self.rollout_size, self.row_dim, pin_memory=True)
        self.staging_np = self.staging.numpy()
        self.fresh = torch.zeros(self.rollout_size, self.row_dim, device=device)  # the latest rollout

        f32 = dict(device=device, dtype=torch.float32)
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.clipping_statistics = torch.zeros(3, 5, **f32)
        # SDN acting sums since the last rollout boundary: selected-minus-mean candidate Q,
        # within-state candidate Q std, selected rank, and the number of env rows.
        self.sdn_statistics = torch.zeros(4, **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "smoothing_gap", "preaction_gap",
            *DBU_DIAGNOSTICS, "sigma_floor_fraction", "sigma_ceiling_fraction",
            "sigma_geometric_mean", "entropy_target_error",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._actor_loss = maybe_compile(self._actor_loss_fn)
        self._entropy = maybe_compile(self._entropy_fn)
        # Replay mode: training_steps % policy_freq == 0 selects the actor-step graph (TD7's delayed update).
        steps = [self._make_step(update_actor=False)]
        steps.append(self._make_step(update_actor=True))
        self.graphs = capture_steps(
            steps,
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics, self.clipping_statistics],
        )
        if args.sdn_candidates:
            self.sdn_obs = torch.zeros(args.num_envs, state_dim, **f32)
            # action | mu | sigma | eps | pre_action of the executed candidate, one row per env.
            self.sdn_out = torch.zeros(args.num_envs, 5 * action_dim, **f32)
            self.sdn_cumulative = rank_selection_probabilities(
                args.sdn_candidates, args.sdn_rank_temperature).cumsum(0).to(device)
            self.sdn_graph, = capture_steps([self._sdn_step], optimizers=[],
                                            state_tensors=[self.sdn_out, self.sdn_statistics])
            self.sdn_host = torch.zeros(args.num_envs, 5 * action_dim, pin_memory=True)
            self.sdn_obs_host = torch.zeros(args.num_envs, state_dim, pin_memory=True)

        self.policy = HostPolicy(self.fixed_encoder, self.actor)
        self.policy.refresh()
        self.training_steps = 0

    def checkpoint_state(self):
        """Complete learner state at a completed rollout; save only filled replay rows."""
        return checkpoint_to_cpu({
            "models": {name: getattr(self, name).state_dict() for name in CHECKPOINT_MODELS},
            "optimizers": {name: getattr(self, name).state_dict() for name in CHECKPOINT_OPTIMIZERS},
            "log_alpha": self.log_alpha,
            "buffers": {name: getattr(self, name) for name in CHECKPOINT_BUFFERS},
            "storage": self.storage[:self.size], "priority": self.priority[:self.size],
            "ptr": self.ptr, "size": self.size, "training_steps": self.training_steps,
            "current_global_step": getattr(self, "current_global_step", 0),
        })

    def restore_checkpoint_state(self, state):
        """Copy into captured parameters, replay buffers and Adam state in place."""
        size, ptr = int(state["size"]), int(state["ptr"])
        if not 0 <= size <= self.args.buffer_size or not 0 <= ptr < self.args.buffer_size:
            raise ValueError("invalid checkpoint replay cursors")
        if state["storage"].shape != (size, self.row_dim) or state["priority"].shape != (size,):
            raise ValueError("checkpoint replay dimensions differ")
        for name in CHECKPOINT_MODELS:
            getattr(self, name).load_state_dict(state["models"][name], strict=True, assign=False)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
            self.storage.zero_()
            self.storage[:size].copy_(state["storage"])
            self.priority.zero_()
            self.priority[:size].copy_(state["priority"])
            for name in CHECKPOINT_BUFFERS:
                current, previous = getattr(self, name), state["buffers"][name]
                if current.shape != previous.shape or current.dtype != previous.dtype:
                    raise ValueError(f"checkpoint buffer differs: {name}")
                current.copy_(previous)
        for name in CHECKPOINT_OPTIMIZERS:
            restore_optimizer_in_place(getattr(self, name), state["optimizers"][name])
        self.ptr, self.size = ptr, size
        self.training_steps = int(state["training_steps"])
        self.current_global_step = int(state["current_global_step"])
        self.policy.refresh()

    # ---- replay -------------------------------------------------------------------------------
    def stage(self, step, **values):
        """Write one vector-env step: every value is (num_envs, width)."""
        rows = self.staging_np[step * self.args.num_envs : (step + 1) * self.args.num_envs]
        for name, value in values.items():
            rows[:, self._fields[name]] = value

    def flush(self):
        """Upload the staged rollout; new rows get the current max priority, as in LAP's add()."""
        n = self.rollout_size
        self.fresh.copy_(self.staging)
        index = (self.ptr + torch.arange(n, device=self.device)) % self.args.buffer_size
        self.storage.index_copy_(0, index, self.fresh)
        self.priority.index_put_((index,), self.max_priority.expand(n))
        self.ptr = (self.ptr + n) % self.args.buffer_size
        self.size = min(self.size + n, self.args.buffer_size)
        self.last.fill_(self.size - 1)

    # ---- losses -------------------------------------------------------------------------------
    def _encoder_loss_fn(self, batch):
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        return F.mse_loss(pred_zs, next_zs)

    def _critic_loss_fn(self, batch, min_target, max_target):
        f, args = self._fields, self.args
        state, next_state = batch[:, f["state"]], batch[:, f["next_state"]]
        action, mu, sigma, eps = batch[:, f["action"]], batch[:, f["mu"]], batch[:, f["sigma"]], batch[:, f["eps"]]
        pre_action = batch[:, f["pre_action"]]
        with torch.no_grad():
            # Deterministic center query at TD7's target smoothing std; no smoothing action is sampled.
            next_zs = self.fixed_encoder_target.zs(next_state)
            next_mu, _ = self.actor_target(next_state, next_zs)
            next_zsa = self.fixed_encoder_target.zsa(next_zs, next_mu)
            next_latent = torch.cat([next_mu, torch.full_like(next_mu, args.target_smoothing_sigma)], 1)
            q_target = self.critic_target(next_state, next_latent, next_mu, next_zsa, next_zs)
            q_target = q_target.min(1, keepdim=True)[0].clamp(min_target, max_target)
            q_target = batch[:, f["reward"]] + batch[:, f["not_done"]] * args.gamma * q_target

            # Three queries share one critic and one 3B pass: distribution center (mu, sigma, mu)
            # with Hermite tangent, observed pre-action latent (mu, sigma, u), and point action
            # (a, 0, a). The latter two get direct TD labels; center and point coincide at sigma=0.
            fixed_zs = self.fixed_encoder.zs(state)
            t_mu = sigma * eps if args.hermite_order >= 1 else torch.zeros_like(mu)
            t_sigma = 0.5 * sigma * (eps.square() - 1) if args.hermite_order >= 2 else torch.zeros_like(sigma)
            zs3 = torch.cat([fixed_zs, fixed_zs, fixed_zs])
            zsa3, t_zsa3 = self.fixed_encoder.zsa_jvp(
                zs3, torch.cat([mu, action, action]), torch.cat([t_mu, torch.zeros_like(action), torch.zeros_like(action)])
            )
            latent3 = torch.cat([
                torch.cat([mu, sigma], 1),
                torch.cat([mu, sigma], 1),
                torch.cat([action, torch.zeros_like(sigma)], 1),
            ])
            pre_actions3 = torch.cat([mu, pre_action, action])
            t_pre_actions3 = torch.cat([t_mu, torch.zeros_like(action), torch.zeros_like(action)])
            t_latent3 = torch.cat([
                torch.cat([t_mu, t_sigma], 1),
                torch.zeros_like(latent3[: len(mu)]),
                torch.zeros_like(latent3[: len(mu)]),
            ])
        states3 = torch.cat([state, state, state])
        q3, tangent3 = self.critic(
            states3, latent3, pre_actions3, zsa3, zs3, t_latent3, t_pre_actions3, t_zsa3
        )
        batch_size = state.shape[0]
        q_dist, tangent = q3[:batch_size], tangent3[:batch_size]
        q_preaction, q_action = q3[batch_size : 2 * batch_size], q3[2 * batch_size :]
        td_latent = (q_dist + tangent - q_target).abs()
        td_preaction = (q_preaction - q_target).abs()
        td_action = (q_action - q_target).abs()
        loss = (
            lap_huber(td_latent, args.min_priority)
            + lap_huber(td_preaction, args.min_priority)
            + lap_huber(td_action, args.min_priority)
        ) / 3
        with torch.no_grad():
            # Fraction of the realized-noise variation in y that the Hermite terms explain.
            r2 = 1 - td_latent.square().mean() / (q_target - q_dist).square().mean().clamp(min=1e-8)
            smoothing_gap = (q_action - q_dist).mean()
            preaction_gap = (q_preaction - q_action).mean()
            diagnostics = torch.stack([
                q_dist.mean(), r2, tangent.std(), q_target.max(), q_target.min(), smoothing_gap, preaction_gap
            ])
        td = torch.stack([td_latent, td_preaction, td_action], dim=0).amax(0)
        return loss, td.detach(), diagnostics, fixed_zs

    def _point_q(self, state, fixed_zs, actions):
        """Twin-mean point head Q(s, a, 0, a) at (count, batch, action) clipped actions."""
        count, batch_size, action_dim = actions.shape
        flat_action = actions.reshape(-1, action_dim)
        states = state.expand(count, -1, -1).reshape(-1, self.state_dim)
        zs = fixed_zs.expand(count, -1, -1).reshape(-1, fixed_zs.shape[-1])
        latent = torch.cat([flat_action, torch.zeros_like(flat_action)], 1)
        return self.critic(states, latent, flat_action,
                           self.fixed_encoder.zsa(zs, flat_action), zs).mean(-1).reshape(count, batch_size)

    def _actor_loss_fn(self, state, fixed_zs, alpha, eps):
        args = self.args
        mu, log_std = self.actor(state, fixed_zs)
        with torch.no_grad():
            sigma = log_std.exp()
            draws = mu + sigma * eps
            candidates = torch.cat([draws, mu[None]]) if args.dbu_include_mode else draws
            q = self._point_q(state, fixed_zs, candidates.clamp(-1, 1))
            target, variance, diagnostics = dbu_targets(candidates, q, len(draws), args.dbu_temperature,
                                                        args.dbu_include_mode)
            diagnostics = torch.cat([diagnostics, torch.stack([
                ((target - mu) / sigma).square().mean(-1).sqrt().mean(),
                (0.5 * variance.clamp(min=1e-12).log() - log_std).mean().exp(),
                (draws.abs() > 1).float().mean(),
                (log_std < LOG_STD_MIN + 0.05).float().mean(),
                (log_std > LOG_STD_MAX - 0.05).float().mean(),
                log_std.mean().exp(),
                math.log(args.target_sigma) - log_std.mean(),
            ])])
        location, scale = dbu_losses(mu, log_std, target, variance)
        return location + scale - alpha * log_std.sum(-1).mean(), log_std, diagnostics

    def _sdn_step(self):
        """SDN acting: K Gaussian candidates per env, executed by rank softmax over point-Q."""
        args = self.args
        with torch.no_grad():
            state = self.sdn_obs
            fixed_zs = self.fixed_encoder.zs(state)
            mu, log_std = self.actor(state, fixed_zs)
            sigma = log_std.exp()
            eps = torch.randn(args.sdn_candidates, args.num_envs, self.action_dim, device=self.device)
            pre_action = mu + sigma * eps
            action = pre_action.clamp(-1, 1)
            q = self._point_q(state, fixed_zs, action)
            order = q.argsort(0, descending=True)
            draw = torch.rand(args.num_envs, device=self.device) * self.sdn_cumulative[-1]
            rank = torch.searchsorted(self.sdn_cumulative, draw).clamp(max=args.sdn_candidates - 1)
            index = order.gather(0, rank[None])
            pick = index[..., None].expand(1, args.num_envs, self.action_dim)
            self.sdn_out.copy_(torch.cat([
                action.gather(0, pick)[0], mu, sigma, eps.gather(0, pick)[0], pre_action.gather(0, pick)[0],
            ], 1))
            self.sdn_statistics.add_(torch.stack([
                (q.gather(0, index)[0] - q.mean(0)).sum(), q.std(0).sum(),
                rank.sum().float(), q.new_full((), float(args.num_envs)),
            ]))

    def act(self, obs):
        """Replay the SDN graph on host observations; returns host views overwritten next call."""
        self.sdn_obs_host.numpy()[:] = obs
        self.sdn_obs.copy_(self.sdn_obs_host, non_blocking=True)
        self.sdn_graph.replay()
        self.sdn_host.copy_(self.sdn_out, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return np.split(self.sdn_host.numpy(), 5, axis=1)

    def _entropy_fn(self, state, fixed_zs):
        with torch.no_grad():
            return self.actor(state, fixed_zs)[1].sum(-1)

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, device = self.args, self.device

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = torch.searchsorted(csum, torch.rand(args.batch_size, device=device) * csum.gather(0, self.last))
            ind = torch.minimum(ind, self.last)  # the scan is only monotone up to rounding
            batch = self.storage.index_select(0, ind)

            self.encoder_optimizer.zero_grad(set_to_none=False)
            encoder_loss = self._encoder_loss(batch)
            encoder_loss.backward()
            norm = nn.utils.clip_grad_norm_(self.encoder.parameters(), args.max_grad_norm)
            accumulate_clipping_statistics(self.clipping_statistics[0], norm, args.max_grad_norm)
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            critic_loss, td, diagnostics, fixed_zs = self._critic_loss(batch, self.min_target, self.max_target)
            critic_loss.backward()
            norm = nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            accumulate_clipping_statistics(self.clipping_statistics[1], norm, args.max_grad_norm)
            self.critic_optimizer.step()

            with torch.no_grad():
                torch.maximum(self.running_max, diagnostics[3], out=self.running_max)
                torch.minimum(self.running_min, diagnostics[4], out=self.running_min)
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                self.metrics[2:5].copy_(diagnostics[:3])
                self.metrics[9:11].copy_(diagnostics[5:7])

            if update_actor:
                state = batch[:, self._fields["state"]]
                self.actor_optimizer.zero_grad(set_to_none=False)
                eps = torch.randn(args.dbu_samples // 2, args.batch_size, self.action_dim, device=device)
                eps = torch.cat([eps, -eps])
                actor_loss, log_std, actor_diagnostics = self._actor_loss(
                    state, fixed_zs, effective_entropy_alpha(self.log_alpha, args.entropy_floor), eps)
                actor_loss.backward(inputs=list(self.actor.parameters()))
                norm = nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
                accumulate_clipping_statistics(self.clipping_statistics[2], norm, args.max_grad_norm)
                self.actor_optimizer.step()
                if args.entropy_floor:
                    entropy = self._entropy(state, fixed_zs)
                    self.alpha_optimizer.zero_grad(set_to_none=False)
                    alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
                    self.metrics[11:].copy_(actor_diagnostics)
                    self.metrics[5:9].copy_(torch.stack([actor_loss.detach(), sigma.mean(), sigma.min(), sigma.max()]))

        return step

    def hard_update(self):
        with torch.no_grad():
            for destination, source in self._hard_updates:
                torch._foreach_copy_(destination, source)
            self.max_priority.copy_(self.priority.max())
            self.max_target.copy_(self.running_max)
            self.min_target.copy_(self.running_min)

    def train(self):
        self.training_steps += 1
        update_actor = self.training_steps % self.args.policy_freq == 0
        self.graphs[int(update_actor)].replay()
        if self.training_steps % self.args.target_update_rate == 0:
            self.hard_update()
        if self.training_steps % 500 == 0:
            # One batched transfer covers every update since the last interval.
            clipping = self.clipping_statistics.tolist()
            self.clipping_statistics.zero_()
            for name, (norm_sum, count, clipped_count, multiplier_sum, norm_max) in zip(
                    ("encoder", "critic", "actor"), clipping):
                if count:
                    for metric, value in (("preclip_norm_mean", norm_sum / count),
                                          ("preclip_norm_max", norm_max),
                                          ("clipped_fraction", clipped_count / count),
                                          ("clip_multiplier_mean", multiplier_sum / count),
                                          ("update_count", count)):
                        self.writer.add_scalar(f"gradients/{name}/{metric}", value, self.training_steps)
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std", "smoothing_gap", "preaction_gap"):
                self.writer.add_scalar(f"losses/{name}", values[name], self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.training_steps)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.training_steps)
                self.writer.add_scalar("losses/actor_loss", values["actor_loss"], self.training_steps)
                self.writer.add_scalar("losses/alpha", effective_entropy_alpha(self.log_alpha, self.args.entropy_floor).item(), self.training_steps)
                for name in self.metric_names[11:]:
                    self.writer.add_scalar(f"dbu/{name}", values[name], self.current_global_step)

    # ---- one rollout of learning -------------------------------------------------------------
    def learn(self, global_step):
        self.current_global_step = global_step
        state = self.fresh[:, self._fields["state"]]
        with torch.no_grad():
            mu_old, log_std_old = self.actor(state, self.fixed_encoder.zs(state))
        for _ in range(self.updates_per_rollout):
            self.train()
        with torch.no_grad():
            mu, log_std = self.actor(state, self.fixed_encoder.zs(state))
            mean_term, scale_term = gaussian_kl(mu_old, log_std_old.exp(), mu, log_std)
            stats = torch.stack([(mean_term + scale_term).mean(), mean_term.mean(), scale_term.mean()]).tolist()
        for tag, stat in zip(("actor/kl", "actor/kl_mean_term", "actor/kl_scale_term"), stats):
            self.writer.add_scalar(tag, stat, global_step)
        if self.args.sdn_candidates:
            selected_gap, spread, rank, rows = self.sdn_statistics.tolist()
            self.sdn_statistics.zero_()
            if rows:
                for tag, value in (("selected_q_gap", selected_gap / rows), ("candidate_q_std", spread / rows),
                                   ("selected_rank", rank / rows)):
                    self.writer.add_scalar(f"sdn/{tag}", value, global_step)
        self.policy.refresh()


def transition_observations(next_obs, boundaries, infos):
    """next_obs with autoreset rows replaced by the finished episode's final observation."""
    if not boundaries.any():
        return next_obs
    transition_obs = next_obs.copy()
    for index in np.flatnonzero(boundaries):
        transition_obs[index] = infos["final_observation"][index]
    return transition_obs


def evaluate(policy: HostPolicy, eval_envs, max_action):
    """Deterministic policy returns (the latent mean), one episode per env, in lockstep."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        actions, _ = policy(states)
        for i in np.flatnonzero(active):
            state, reward, terminated, truncated, _ = eval_envs[i].step(actions[i] * max_action)
            states[i] = state
            returns[i] += reward
            active[i] = not (terminated or truncated)
    return returns.mean()


def main():
    args = tyro.cli(Args)
    checkpoint_cadence = CheckpointCadence(args.checkpoint_interval_seconds)
    # Resume files are this trainer's own trusted torch.save artifacts.
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint is not None:
        validate_checkpoint_args(checkpoint, args)
    resuming = checkpoint is not None
    resumed_global_step = int(checkpoint["global_step"]) if resuming else 0
    resumed_rollouts = int(checkpoint["completed_rollouts"]) if resuming else 0
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_path = Path(args.checkpoint_path or f"runs/{run_name}/checkpoint.pt")
    stop_requested = False

    def request_shutdown(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    with ExitStack() as cleanup:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
            cleanup.callback(signal.signal, signum, previous)
        writer = SummaryWriter(f"runs/{run_name}")
        cleanup.callback(writer.close)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        configure_runtime()
        assert torch.cuda.is_available(), "CUDA is required"
        device = torch.device("cuda")

        envs = make_mujoco_vector_env(args.env_id, args.num_envs, num_threads=args.env_threads)
        cleanup.callback(envs.close)
        eval_envs = [gym.make(args.env_id) for _ in range(args.eval_eps)]
        for eval_env in eval_envs:
            cleanup.callback(eval_env.close)
        for i, eval_env in enumerate(eval_envs):
            eval_env.reset(seed=args.seed + 100 + i)
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        state_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        max_action = float(envs.single_action_space.high[0])
        shape = (args.num_envs, action_dim)

        agent = TD7Agent(state_dim, action_dim, args, device, writer)
        if checkpoint is not None:
            agent.restore_checkpoint_state(checkpoint["agent"])
            restore_rng_state(checkpoint["rng"], rng)
            writer.add_text("resumed_from", args.resume)
            print(f"resumed_from={args.resume}, global_step={checkpoint['global_step']}; "
                  "fresh MuJoCo trajectories and stochastic policy phase warmup", flush=True)
        # All restored tensors now live in existing device buffers. Release the
        # large CPU replay snapshot instead of retaining it during training.
        checkpoint = None
        timer = PhaseTimer()
        warmup_mu = np.zeros(shape, dtype=np.float32)
        warmup_sigma = np.full(shape, args.warmup_sigma, dtype=np.float32)

        start_time = time.time()
        # Resume preserves learner/replay/RNG state, but restarts MuJoCo trajectories.
        # New stochastic phase warmup consumes the continued host RNG; it is not
        # recorded in replay or charged again against the resumed timestep budget.
        horizon = episode_horizon(args.env_id)
        phase_offsets = compute_phase_offsets(args.num_envs, horizon, args.seed)
        writer.add_text("initial_phase_offsets", ",".join(map(str, phase_offsets)))

        def phase_actions(state):
            if resuming:
                mu, log_std = agent.policy(state)
                sigma = np.exp(log_std)
                return np.clip(mu + sigma * rng.standard_normal(shape, dtype=np.float32), -1, 1) * max_action
            # Preserve v1's fresh-run phase warmup exactly.
            return np.clip(args.warmup_sigma * rng.standard_normal(shape, dtype=np.float32), -1, 1) * max_action

        warm = run_phase_warmup(
            envs, obs_norm=None, horizon=horizon, phase_offsets=phase_offsets, seed=args.seed,
            act_fn=phase_actions,
        )
        obs = warm.next_obs
        global_step = resumed_global_step if resuming else warm.transitions
        initial_global_step = resumed_global_step
        completed_rollouts = resumed_rollouts
        suppress = warm.suppress_mask.copy()
        remaining_rollouts = max(0, (args.total_timesteps - global_step) // agent.rollout_size)
        num_iterations = completed_rollouts + remaining_rollouts
        for iteration in range(completed_rollouts + 1, num_iterations + 1):
            if stop_requested:
                break
            training = iteration > args.warmup_rollouts
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    if training and args.sdn_candidates:
                        action, mu, sigma, eps, pre_action = agent.act(obs)
                    else:
                        if training:
                            mu, log_std = agent.policy(obs)
                            sigma = np.exp(log_std)
                        else:
                            mu, sigma = warmup_mu, warmup_sigma
                        eps = rng.standard_normal(shape, dtype=np.float32)
                        pre_action = mu + sigma * eps
                        action = np.clip(pre_action, -1, 1)
                with timer.span("env", use_cuda=False):
                    next_obs, reward, terminated, truncated, infos = envs.step(action * max_action)
                with timer.span("rollout", use_cuda=False):
                    next_obs = next_obs.astype(np.float32)
                    agent.stage(
                        step, state=obs, action=action, mu=mu, sigma=sigma, eps=eps, pre_action=pre_action,
                        next_state=transition_observations(next_obs, terminated | truncated, infos),
                        reward=reward[:, None], not_done=1.0 - terminated[:, None],
                    )
                    obs = next_obs
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return:.3f}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("update"):
                agent.flush()
                if training:
                    agent.learn(global_step)
            completed_rollouts = iteration
            # SIGTERM/SIGINT request a checkpoint at a complete rollout. A SIGKILL,
            # or a queue grace shorter than this burst, leaves the last periodic file.
            if stop_requested:
                break
            if iteration % args.eval_interval == 0 or iteration == num_iterations:
                with timer.span("eval", use_cuda=False):
                    eval_return = evaluate(agent.policy, eval_envs, max_action)
                writer.add_scalar("eval/episodic_return", eval_return, global_step)
                print(f"global_step={global_step}, eval_return={eval_return:.3f}")
            writer.add_scalar("charts/SPS", int((global_step - initial_global_step) / (time.time() - start_time)), global_step)
            for name, stats in timer.summary().items():
                writer.add_scalar(f"time/{name}_s", stats["total_s"], global_step)
            timer.reset()
            if checkpoint_cadence.periodic_due():
                writer.flush()
                save_checkpoint(checkpoint_path, agent, rng, global_step, completed_rollouts)
                checkpoint_cadence.record_commit((global_step, completed_rollouts, agent.training_steps))

        terminal_state = (global_step, completed_rollouts, agent.training_steps)
        if checkpoint_cadence.needs_terminal_commit(terminal_state):
            writer.flush()
            save_checkpoint(checkpoint_path, agent, rng, global_step, completed_rollouts)
            checkpoint_cadence.record_commit(terminal_state)
        if stop_requested:
            print("shutdown requested; saved completed-rollout checkpoint", flush=True)


if __name__ == "__main__":
    main()
