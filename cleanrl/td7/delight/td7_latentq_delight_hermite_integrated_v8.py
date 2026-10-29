# TD7 LATENT-Q INTEGRATED HERMITE DELIGHT v8
# Integrate the ordinary Hermite Gaussian-score gradient exactly, then sample
# only the delight-minus-PG correction using the same four antithetic actions.
# Frozen center derivatives include mean, pre-action and SALE embedding routes;
# no actor gradient traverses the critic and no point-Q teacher is introduced.
# This preserves the expected native DG gradient while removing base-PG noise.
# Retain entropy tuning; default to zero target smoothing after the v7 result.
# Record every preclip gradient norm and clipping multiplier on the device.
# Point-Q actor mode remains available as a matched v5 control.
import os
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
    """resume a v8 rollout-boundary checkpoint; MuJoCo trajectories restart"""
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
    actor_objective: str = "delight"
    """delight: gated score; pg: ungated score control; pathwise: v8 center-Q control"""
    actor_advantage: str = "hermite"
    """hermite: native center geometry; point: v5 point-Q baseline control"""
    actor_baseline: str = "latent"
    """latent: learned center V; cubature: original v1 baseline control"""
    actor_samples: int = 4
    """even number of current-policy antithetic score samples per replay state"""
    delight_temperature: float = 1.0
    """gate temperature; large values recover the ungated score update"""
    surprisal: str = "relative"
    """relative: Gaussian mode-relative rarity; density: signed latent log-density ablation"""
    surprisal_cap: float = 10.0
    """paper's surprisal cap; relative uses [0,cap], density uses [-cap,cap]"""
    gate_scale: str = "rms"
    """rms: divide gate advantages by detached RMS; raw: retain raw reward units"""
    gate_scope: str = "mean"
    """mean: gate location score only; all: also gate scale score (paper-style ablation)"""
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
    """retain entropy bonus and alpha tuner; false leaves only native scale PG"""
    target_sigma: float = 0.1
    """minimum mean log-std entropy target for the positive alpha tuner; not an upper sigma bound"""
    target_smoothing_sigma: float = 0.0
    """per-dim std of Gaussian target smoothing; zero follows the v7 ablation"""
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


def gaussian_cubature(action_dim, *, device=None, dtype=torch.float32):
    """Equal-weight 2d rule, exact through degree three under a standard Gaussian."""
    axes = torch.eye(action_dim, device=device, dtype=dtype) * action_dim**0.5
    return torch.cat([axes, -axes])


def gaussian_log_probability(pre_action, mu, log_std):
    """Joint latent density at FIXED u, before physical action clipping."""
    standardized = (pre_action.detach() - mu) * (-log_std).exp()
    return (-0.5 * standardized.square() - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def delight_gate(pre_action, mu, log_std, advantage, *, objective="delight",
                 temperature=1.0, surprisal="relative", cap=10.0, scale="rms"):
    """Frozen gate and the original eight native-DG diagnostics."""
    with torch.no_grad():
        advantage = advantage.detach()
        if surprisal == "relative":
            eps = (pre_action.detach() - mu.detach()) * (-log_std.detach()).exp()
            rarity = (0.5 * eps.square().sum(-1)).clamp(max=cap)
        else:
            rarity = (-gaussian_log_probability(pre_action, mu, log_std)).clamp(-cap, cap)
        rms = advantage.square().mean().sqrt()
        gate_units = rms.clamp(min=1e-8) if scale == "rms" else torch.ones_like(rms)
        weight = (2 * torch.sigmoid(advantage / gate_units * rarity / temperature)
                  if objective == "delight" else torch.ones_like(advantage))
        diagnostics = torch.stack([
            weight.mean(), (weight < 0.02).float().mean(), (weight > 1.98).float().mean(),
            advantage.mean(), rms, rarity.mean(), (rarity < 0).float().mean(),
            (pre_action.detach().abs() > 1).float().mean(),
        ])
    return weight, diagnostics


def delight_score_loss(pre_action, mu, log_std, advantage, *, objective="delight",
                       temperature=1.0, surprisal="relative", cap=10.0, scale="rms", scope="mean"):
    """Only the score is differentiated; advantages/gates have no gradient path.

    Inputs are (samples,batch,action), (batch,action), and (samples,batch).
    Antithetic samples share a deterministic baseline, never each other's Q.
    The diagnostics are estimates, not a certificate of cubature accuracy.
    """
    log_prob = gaussian_log_probability(pre_action, mu, log_std)
    weight, diagnostics = delight_gate(
        pre_action, mu, log_std, advantage, objective=objective, temperature=temperature,
        surprisal=surprisal, cap=cap, scale=scale,
    )
    advantage = advantage.detach()
    if scope == "mean":
        # Full PG plus a location-only correction. Frozen std removes the scale
        # gradient from the correction, including shared-trunk contributions.
        location_log_prob = gaussian_log_probability(pre_action, mu, log_std.detach())
        surrogate = advantage * (log_prob + (weight - 1) * location_log_prob)
    else:
        surrogate = weight * advantage * log_prob
    return -surrogate.mean(), diagnostics


def integrated_hermite_score_loss(pre_action, mu, log_std, advantage, grad_mu, grad_sigma,
                                  *, order=2, objective="delight", temperature=1.0,
                                  surprisal="relative", cap=10.0, scale="rms", scope="mean"):
    """Exact ordinary score expectation plus the unchanged sampled DG correction.

    E[H score_mu] = grad_mu; E[H score_logsigma] = sigma * grad_sigma.
    Coefficients are frozen, so the actor never differentiates through V.
    Sample-derived gate normalization remains unchanged in the correction.
    """
    mean_coefficient = grad_mu.detach()
    scale_coefficient = ((log_std.detach().exp() * grad_sigma.detach())
                         if order >= 2 else torch.zeros_like(log_std))
    loss = -(mu * mean_coefficient + log_std * scale_coefficient).sum(-1).mean()
    weight, diagnostics = delight_gate(
        pre_action, mu, log_std, advantage, objective=objective, temperature=temperature,
        surprisal=surprisal, cap=cap, scale=scale,
    )
    if objective == "delight":
        correction_log_prob = gaussian_log_probability(
            pre_action, mu, log_std.detach() if scope == "mean" else log_std,
        )
        loss = loss - ((weight - 1) * advantage.detach() * correction_log_prob).mean()
    return loss, diagnostics


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


def value_baseline_advantage(sampled_q, latent_v, cubature_v, baseline):
    """Detach the physical action value and the chosen action-independent baseline."""
    value = latent_v if baseline == "latent" else cubature_v
    return sampled_q.detach() - value.detach()


def hermite_advantage(eps, sigma, grad_mu, grad_sigma, order=2):
    """Centered Gaussian Hermite advantage; every coefficient is score-detached."""
    eps, sigma = eps.detach(), sigma.detach()
    first = (sigma * eps * grad_mu.detach()).sum(-1)
    second = (0.5 * sigma * (eps.square() - 1) * grad_sigma.detach()).sum(-1)
    if order == 1:
        second = torch.zeros_like(first)
    return first + second, first, second


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
                      "max_target", "min_target", "metrics", "fresh", "clipping_statistics")
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
    if checkpoint.get("format") != "td7_latentq_delight_hermite_integrated_v8" or checkpoint.get("version") != 1:
        raise ValueError("expected a integrated Hermite v8 training checkpoint")
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
    checkpoint = {"format": "td7_latentq_delight_hermite_integrated_v8", "version": 1,
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
        if args.actor_baseline not in ("latent", "cubature"):
            raise ValueError("actor_baseline must be latent or cubature")
        if args.actor_objective not in ("delight", "pg", "pathwise"):
            raise ValueError("actor_objective must be delight, pg or pathwise")
        if args.actor_samples < 2 or args.actor_samples % 2:
            raise ValueError("actor_samples must be a positive even number >= 2")
        if not np.isfinite(args.delight_temperature) or args.delight_temperature <= 0:
            raise ValueError("delight_temperature must be finite and positive")
        if args.surprisal not in ("relative", "density") or args.gate_scale not in ("rms", "raw"):
            raise ValueError("invalid surprisal or gate_scale")
        if args.gate_scope not in ("mean", "all"):
            raise ValueError("gate_scope must be mean or all")
        if not np.isfinite(args.surprisal_cap) or args.surprisal_cap <= 0:
            raise ValueError("surprisal_cap must be finite and positive")
        if args.actor_advantage not in ("hermite", "point"):
            raise ValueError("actor_advantage must be hermite or point")
        if args.actor_advantage == "hermite" and args.hermite_order == 0:
            raise ValueError("native Hermite policy requires hermite_order >= 1")
        if args.actor_advantage == "hermite" and args.actor_objective == "pathwise":
            raise ValueError("native Hermite policy uses delight or pg scores; pathwise requires point mode")
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim if args.actor_sale else 0
        self.actor = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.cubature = gaussian_cubature(action_dim, device=device)
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
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
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
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "hermite_r2", "tangent_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "smoothing_gap", "preaction_gap",
            "gate_mean", "gate_suppressed", "gate_saturated", "advantage_mean", "advantage_rms",
            "surprisal_mean", "negative_surprisal_fraction", "clipped_action_fraction",
            "cubature_center_gap", "baseline_gap_rms", "baseline_sign_disagreement",
            "latent_v_mean", "cubature_v_mean", "positive_advantage_fraction",
        ]
        if args.actor_advantage == "hermite":
            self.metric_names[19:25] = [
                "hermite_center_mean", "hermite_first_rms", "hermite_second_rms",
                "hermite_analytic_rms", "hermite_pair_coefficient_cosine", "hermite_pair_reversal_fraction",
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

    def _center_value_fn(self, mu, sigma, state, fixed_zs):
        # One mean variable enters all three routes, so differentiation is total:
        # latent mean, pre-action center, and the action-conditioned embedding.
        center = self.critic(state, torch.cat([mu, sigma], 1), mu,
                             self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs).mean(-1)
        return center.sum(), center

    def _hermite_actor_loss_fn(self, state, fixed_zs, alpha):
        args = self.args
        mu, log_std = self.actor(state, fixed_zs)
        with torch.no_grad():
            sigma = log_std.exp()
            (grad_mu, grad_sigma), center = torch.func.grad(
                self._center_value_fn, argnums=(0, 1), has_aux=True,
            )(mu.detach(), sigma.detach(), state.detach(), fixed_zs.detach())
            eps = torch.randn(args.actor_samples // 2, *mu.shape, device=state.device)
            eps = torch.cat([eps, -eps])
            sampled_u = mu + sigma * eps
            advantage, first, second = hermite_advantage(eps, sigma, grad_mu, grad_sigma, args.hermite_order)
        score_loss, diagnostics = integrated_hermite_score_loss(
            sampled_u, mu, log_std, advantage, grad_mu, grad_sigma,
            order=args.hermite_order, objective=args.actor_objective,
            temperature=args.delight_temperature, surprisal=args.surprisal,
            cap=args.surprisal_cap, scale=args.gate_scale, scope=args.gate_scope,
        )
        with torch.no_grad():
            rarity = ((0.5 * eps.square().sum(-1)).clamp(max=args.surprisal_cap)
                      if args.surprisal == "relative" else
                      (-gaussian_log_probability(sampled_u, mu, log_std)).clamp(-args.surprisal_cap, args.surprisal_cap))
            units = diagnostics[4].clamp(min=1e-8) if args.gate_scale == "rms" else torch.ones_like(diagnostics[4])
            weight = (2 * torch.sigmoid(advantage / units * rarity / args.delight_temperature)
                      if args.actor_objective == "delight" else torch.ones_like(advantage))
            half = args.actor_samples // 2
            ordinary_pair = advantage[:half] - advantage[half:]
            gated_pair = weight[:half] * advantage[:half] - weight[half:] * advantage[half:]
            norm = (ordinary_pair.square().sum(0) * gated_pair.square().sum(0)).sqrt().clamp(min=1e-8)
            cosine = (ordinary_pair * gated_pair).sum(0) / norm
            second_variance = (0.5 * (sigma * grad_sigma.detach()).square().sum(-1)
                               if args.hermite_order >= 2 else torch.zeros_like(center))
            analytic_rms = ((sigma * grad_mu.detach()).square().sum(-1) + second_variance).mean().sqrt()
            native_diagnostics = torch.stack([
                center.mean(), first.square().mean().sqrt(), second.square().mean().sqrt(),
                analytic_rms, cosine.mean(), (ordinary_pair * gated_pair < 0).float().mean(),
            ])
        return (score_loss - alpha * log_std.sum(-1).mean(), log_std,
                torch.cat([diagnostics, native_diagnostics]))

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        args = self.args
        if not args.entropy_floor:
            alpha = torch.zeros_like(alpha)
        if args.actor_advantage == "hermite":
            return self._hermite_actor_loss_fn(state, fixed_zs, alpha)
        mu, log_std = self.actor(state, fixed_zs)
        if args.actor_objective == "pathwise":
            latent = torch.cat([mu, log_std.exp()], 1)
            q = self.critic(state, latent, mu, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs)
            return (-q.mean() - alpha * log_std.sum(-1).mean(), log_std,
                    torch.zeros(14, device=state.device))
        with torch.no_grad():
            eps = torch.randn(args.actor_samples // 2, *mu.shape, device=state.device)
            eps = torch.cat([eps, -eps])
            sampled_u = mu + log_std.exp() * eps
            baseline_u = mu + log_std.exp() * self.cubature[:, None, :]
            u = torch.cat([sampled_u, baseline_u])
            samples, batch_size, action_dim = u.shape
            flat_action = u.clamp(-1, 1).reshape(-1, action_dim)
            states = state.expand(samples, -1, -1).reshape(-1, self.state_dim)
            zs = fixed_zs.expand(samples, -1, -1).reshape(-1, args.zs_dim)
            # Point head: Q(s,a), independent of the sampled policy's artificial context.
            latent = torch.cat([flat_action, torch.zeros_like(flat_action)], 1)
            q = self.critic(states, latent, flat_action,
                            self.fixed_encoder.zsa(zs, flat_action), zs).mean(-1).reshape(samples, batch_size)
            cubature_v = q[args.actor_samples:].mean(0)
            center_q = self.critic(state, torch.cat([mu, log_std.exp()], 1), mu,
                                   self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs).mean(-1)
            sampled_q = q[:args.actor_samples]
            advantage = value_baseline_advantage(sampled_q, center_q, cubature_v, args.actor_baseline)
            gap = cubature_v - center_q
            sign_disagreement = ((sampled_q > center_q) != (sampled_q > cubature_v)).float().mean()
            value_diagnostics = torch.stack([
                gap.mean(), gap.square().mean().sqrt(), sign_disagreement,
                center_q.mean(), cubature_v.mean(), (advantage > 0).float().mean(),
            ])
        score_loss, diagnostics = delight_score_loss(
            sampled_u, mu, log_std, advantage, objective=args.actor_objective,
            temperature=args.delight_temperature, surprisal=args.surprisal,
            cap=args.surprisal_cap, scale=args.gate_scale, scope=args.gate_scope,
        )
        return (score_loss - alpha * log_std.sum(-1).mean(), log_std,
                torch.cat([diagnostics, value_diagnostics]))

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
                actor_loss, log_std, actor_diagnostics = self._actor_loss(
                    state, fixed_zs, effective_entropy_alpha(self.log_alpha, args.entropy_floor))
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
                    self.writer.add_scalar(f"delight/{name}", values[name], self.current_global_step)

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
