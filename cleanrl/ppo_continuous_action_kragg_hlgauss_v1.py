# Pre-RMS SiTU-GLU PPO with kraggiculture's HL-Gauss numerics, v1.
#
# Mechanism: 255 value logits over a support built by reflection, so
# s[i] == -s[K-1-i] bit-exactly and the centre atom is exactly 0. Labels are
# truncated-Gaussian bin masses over FINITE outer edges (s[0]-w/2, s[-1]+w/2)
# with an erf/erfc branch switch, clamp_min(0) and a reflection-safe pairwise
# renormalization; the value is decoded as a mirrored-pair difference, so a
# symmetric head decodes to exactly the support midpoint with no cancellation.
# Targets are clipped into the support before the label build and the saturated
# fraction is logged (kraggiculture aborts above 5%; this file only logs).
#
# Deviation, stated up front: kraggiculture's support is FIXED and RAW,
# [-2.2, 2.2] return units, with no EMA. That is meaningless here --
# reward-normalized HalfCheetah GAE targets drift from mean 0 to mean ~4 with a
# cross-state spread of 0.25-0.6, so a fixed raw support saturates immediately
# and their own guard would abort the run. This file therefore keeps the
# standardized FRAME of ppo_continuous_action_normres_stdhlgauss_v1 (EMA
# mean/scale of the value targets, frozen for the whole iteration, so
# V = mean + scale*E[z]) and replaces only the numerics inside it. What is
# ablated is kraggiculture's GEOMETRY and label/decode arithmetic, not its
# untransportable choice of absolute support.
#
# Geometry: 255 atoms on z in [-6, 6] target std at sigma = 3.0 bin widths ->
# w = 0.047244 target std (21.17 bins per target std), sigma = 0.141732 target
# std: 1.89x the bandwidth and 2.12x the resolution of the std variant's
# 101-bin / sigma-0.075 grid, and close to kraggiculture's late-run 0.133.
#
# Gradient calibration: dCE/dV = dMSE/dV / Var_p(V), so raw CE runs at
# 1/(scale^2 * (sigma_z^2 + w_z^2/12)) = 49.32/scale^2 times MSE's value-space
# step (the std variant's geometry: 154.84/scale^2). With 1 minibatch x 10
# epochs at lr 9.6e-3 that blows the critic up, so `ce_scale` defaults to
# `matched`, multiplying CE by scale^2*(sigma_z^2 + w_z^2/12) so vf_coef=0.5
# keeps its MSE value-space meaning; `--ce-scale raw` is the faithful sub-arm.
#
# Two further deliberate deviations. The logit softcap 23*sigmoid((z+5)/7.5)
# ships OFF: it fixes critic-gradient growth specific to a deep ViT critic with
# a zeros_ head on an 8.33x LR group behind non-affine norms, and it cost ~6
# waves of warmup there; a 64-unit MLP on one LR group has no such pathology.
# The head keeps layer_init(std=0.01) rather than zeros_, because with only 10
# epochs per iteration a dead first step is unaffordable.
#
# The decoded-value trust region and the {mse, hlgauss, mse_softmax} switch are
# retained so this arm differs from the indclip_v4 scalar baseline in exactly
# one thing; `--value-loss mse --ce-scale raw` is the in-file control.
#
# Hypothesis: the wide bandwidth (3 bins vs 0.75) at 21 bins per target std
# trades label sharpness for robustness to noisy lambda-return targets. If that
# smoothing is what a categorical critic buys, this geometry beats both the
# scalar baseline and the sharper std-HL-Gauss arm at a fixed step budget; if
# resolution is what matters, it should be neutral to worse.
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
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.norm_residual import make_norm_residual_trunk
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
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets,
    episode_horizon,
    run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
SQRT2 = math.sqrt(2.0)
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
# modded-nanogpt's lm_head softcap, as imported by kraggiculture (model.py:47-49).
VALUE_LOGIT_SOFTCAP = 23.0
VALUE_LOGIT_SOFTCAP_SHIFT = 5.0
VALUE_LOGIT_SOFTCAP_WIDTH = 7.5


def categorical_value_support(minimum: float, maximum: float, atoms: int, device=None) -> torch.Tensor:
    """Linear buckets built by mirroring one half, with an exact odd midpoint.

    Transcribed from kraggiculture `model.py:61-69`. This is deliberately not
    `linspace(minimum, maximum, atoms)`: the right half is the reflection of the
    left, so `support[i] == -support[atoms-1-i]` holds bit-exactly and an odd
    atom count puts an atom at exactly the midpoint.
    """
    midpoint = (minimum + maximum) * 0.5
    half = atoms // 2
    step = (maximum - minimum) / (atoms - 1)
    left_end = midpoint if atoms % 2 else midpoint - step * 0.5
    left = torch.linspace(minimum, left_end, half + atoms % 2, dtype=torch.float32, device=device)
    right = minimum + maximum - left[:half].flip(-1)
    return torch.cat((left, right))


def _symmetric_sum(values: torch.Tensor) -> torch.Tensor:
    """Reduce mirrored pairs first so reflection cannot change the normalizer."""
    half = values.shape[-1] // 2
    paired = values[..., :half] + values[..., -half:].flip(-1)
    result = paired.sum(dim=-1)
    return result + values[..., half] if values.shape[-1] % 2 else result


class LogitSoftcap(nn.Module):
    """`23*sigmoid((z+5)/7.5)` on the value readout (kraggiculture model.py:52-58)."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return VALUE_LOGIT_SOFTCAP * torch.sigmoid(
            (logits.float() + VALUE_LOGIT_SOFTCAP_SHIFT) / VALUE_LOGIT_SOFTCAP_WIDTH
        )


class KraggHistogram(nn.Module):
    """kraggiculture's HL-Gauss numerics on a target-standardized support.

    Frame (kept from `shared/hl_gauss_std.py`): the head is a distribution over
    `z`, the raw value is `mean + scale * E[z]`, and `mean`/`scale` are an EMA of
    the value targets frozen for the duration of an iteration.

    Numerics (kraggiculture): reflected support, finite outer edges, erf/erfc
    branch switching, `clamp_min(0)`, reflection-safe renormalization, targets
    clipped into the support, and a cancellation-free mirrored-pair decode.
    """

    centers: torch.Tensor
    edges: torch.Tensor
    upper_offsets: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        atoms: int = 255,
        *,
        half_span: float = 6.0,
        sigma_bins: float = 3.0,
        momentum: float = 0.5,
        min_scale: float = 1e-3,
        device=None,
    ):
        super().__init__()
        if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 3:
            raise ValueError("atoms must be an integer >= 3")
        if not math.isfinite(half_span) or half_span <= 0:
            raise ValueError("half_span must be finite and positive")
        if not math.isfinite(sigma_bins) or sigma_bins <= 0:
            raise ValueError("sigma_bins must be finite and positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        self.atoms = atoms
        self.half_span = float(half_span)
        self.sigma_bins = float(sigma_bins)
        self.momentum = float(momentum)
        self.min_scale = float(min_scale)
        self.half = atoms // 2
        centers = categorical_value_support(-self.half_span, self.half_span, atoms, device=device)
        self.bin_width = float(centers[-1] - centers[0]) / (atoms - 1)
        self.sigma = self.sigma_bins * self.bin_width
        self.midpoint = float(centers[0] + centers[-1]) * 0.5
        # Finite outer edges, not half-infinite: mass past them is dropped and the
        # label renormalized, which is what makes the erfc branch load-bearing.
        interior = (centers[:-1] + centers[1:]) * 0.5
        edges = torch.cat(
            (centers[:1] - self.bin_width * 0.5, interior, centers[-1:] + self.bin_width * 0.5)
        )
        self.register_buffer("centers", centers)
        self.register_buffer("edges", edges)
        self.register_buffer("upper_offsets", centers[-self.half :] - self.midpoint)
        self.register_buffer("mean", torch.zeros((), device=device))
        self.register_buffer("scale", torch.ones((), device=device))
        self.register_buffer("count", torch.zeros((), device=device))
        self._scaled_sigma = SQRT2 * self.sigma

    @torch.no_grad()
    def observe(self, targets: torch.Tensor) -> None:
        """Fold one rollout's value targets into the EMA location and scale.

        The first call adopts the batch statistics outright; the effective
        momentum then ramps to ``self.momentum`` so a cold start is not
        anchored to an uninformative initial value.
        """
        batch_mean = targets.mean()
        batch_scale = targets.std().clamp_min(self.min_scale)
        self.count.add_(1.0)
        weight = torch.minimum(
            torch.full_like(self.count, self.momentum), 1.0 - self.count.reciprocal()
        )
        self.mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.scale.mul_(weight).add_(batch_scale * (1.0 - weight))

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def saturated(self, targets: torch.Tensor) -> torch.Tensor:
        """Fraction of raw targets outside the support, before clipping."""
        return (self.standardize(targets).abs() > self.half_span).float().mean()

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """Gaussian bin masses for raw-unit targets; clipped, clamped, renormalized."""
        z = self.standardize(targets).clamp(-self.half_span, self.half_span)
        u = (self.edges - z.unsqueeze(-1)) / self._scaled_sigma
        lower, upper = u[..., :-1], u[..., 1:]
        erf = torch.erf(u)
        tails = torch.erfc(u.abs())
        central_mass = 0.5 * (erf[..., 1:] - erf[..., :-1])
        tail_mass = 0.5 * (tails[..., 1:] - tails[..., :-1]).abs()
        # erf differences keep narrow central bins; erfc keeps tails where both
        # CDF values round to 1.0. Predicate verbatim from model.py:912.
        central = ((lower <= 0) & (upper >= 0)) | ((lower.abs() < 1) & (upper.abs() < 1))
        mass = torch.where(central, central_mass, tail_mass).clamp_min(0.0)
        return mass / _symmetric_sum(mass).unsqueeze(-1)

    def expectation(self, probs: torch.Tensor) -> torch.Tensor:
        """E_p[z] as a mirrored-pair difference: symmetric probs give exactly the midpoint."""
        paired = probs[..., -self.half :] - probs[..., : self.half].flip(-1)
        return self.midpoint + (paired * self.upper_offsets).sum(dim=-1)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Raw-unit expectation of the categorical head."""
        return self.mean + self.scale * self.expectation(logits.float().softmax(dim=-1))

    def decode_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return self.mean + self.scale * self.expectation(probs)

    def standardized_variance(self, probs: torch.Tensor) -> torch.Tensor:
        """Var_p(z) in standardized units; the inverse CE-vs-MSE gradient gain."""
        first = self.expectation(probs)
        second = (probs * self.centers.square()).sum(dim=-1)
        return (second - first.square()).clamp_min(0.0)

    def mse_equivalent_gain(self) -> torch.Tensor:
        """Constant CE multiplier whose value-space gradient matches 0.5*MSE.

        Uses the converged-head variance ``sigma^2 + bin_width^2/12`` in
        standardized units, times ``scale^2`` to reach raw value units.
        """
        reference = self.sigma**2 + self.bin_width**2 / 12.0
        return self.scale.square() * reference

    def extra_repr(self) -> str:
        return (
            f"atoms={self.atoms}, half_span={self.half_span}, "
            f"sigma_bins={self.sigma_bins}, sigma_z={self.sigma:.6g}, bin_width_z={self.bin_width:.6g}"
        )


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 9.6e-3
    """32x the 3e-4 PPO default; one Adam covers actor and critic"""
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
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """raw GAE in the surrogate; no minibatch standardization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """PPO value trust region; the categorical arms use the decoded-value analogue"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm, applied independently to actor and critic"""
    clip_heads: bool = True
    """include final actor/critic linear heads in their independent clipping budgets;
    True matches the indclip_v4 baseline's clipping budget"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    reward_norm: bool = True
    """normalize/clip rewards before GAE; never normalize GAE return targets"""

    value_loss: Literal["mse", "hlgauss", "mse_softmax"] = "hlgauss"
    """scalar regression, kraggiculture HL-Gauss cross-entropy, or MSE on a K-way head"""
    value_atoms: int = 255
    """categorical head resolution; bin width is 2*half_span/(atoms-1) target std"""
    value_half_span: float = 6.0
    """support half-width in target standard deviations"""
    value_sigma_bins: float = 3.0
    """HL-Gauss sigma as a multiple of the bin width (kraggiculture's promoted value)"""
    value_scale_momentum: float = 0.5
    """EMA retention for the support's location/scale; 32768 targets per rollout
    already estimate them precisely, so heavy smoothing only adds lag"""
    ce_scale: Literal["raw", "matched"] = "matched"
    """`matched` rescales CE so vf_coef keeps its MSE value-space meaning;
    `raw` is kraggiculture-faithful and runs ~49/scale^2 times MSE's value step"""
    ce_sample_weight: Literal["none", "variance"] = "none"
    """`variance` reweights CE per sample to MSE's equal-error weighting"""
    logit_softcap: bool = False
    """bound the value logits by 23*sigmoid((z+5)/7.5); off because the growth
    pathology it fixes belongs to a deep zero-init-head critic, not a 64-unit MLP"""
    critic_width: int = 64
    """critic trunk width; the actor stays at 64"""

    placement: Literal["pre", "post"] = "pre"
    """normalize branch inputs with an identity stream, or residual outputs"""
    norm_kind: Literal["layer", "rms"] = "rms"
    """non-affine centered LayerNorm or uncentered RMSNorm, epsilon 1e-5"""
    activation: Literal["lrelusq", "stiglu"] = "stiglu"
    """squared leaky-ReLU pair or parameter-matched SiTU-GLU branch"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads, capped at num_envs"""
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


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    log_action_scale: torch.Tensor
    histogram: KraggHistogram | None
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs, args):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.categorical = args.value_loss != "mse"
        trunk_kwargs = dict(placement=args.placement, norm_kind=args.norm_kind, activation=args.activation)
        # Trunks and the actor head are built before the critic head so every arm
        # shares one initial policy and one initial critic trunk; only the value
        # readout differs. Small-gain categorical logits keep the initial value at
        # the support center while still giving the critic trunk a gradient on the
        # first step, which kraggiculture's zero-initialized head does not: with 10
        # epochs per iteration a dead first step is unaffordable.
        critic_trunk = make_norm_residual_trunk(observation_dim, args.critic_width, **trunk_kwargs)
        self.actor = nn.Sequential(
            make_norm_residual_trunk(observation_dim, 64, **trunk_kwargs),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        if self.categorical:
            head = layer_init(nn.Linear(args.critic_width, args.value_atoms), std=0.01)
            self.histogram = KraggHistogram(
                args.value_atoms,
                half_span=args.value_half_span,
                sigma_bins=args.value_sigma_bins,
                momentum=args.value_scale_momentum,
            )
        else:
            head = layer_init(nn.Linear(args.critic_width, 1), std=1.0)
            self.histogram = None
        # The softcap sits inside the readout so cross-entropy and decode see the
        # same bounded logits, and `critic[0]` stays the trunk for gradient clipping.
        readout = (critic_trunk, head, LogitSoftcap()) if args.logit_softcap else (critic_trunk, head)
        self.critic = nn.Sequential(*readout)

    def decode(self, readout):
        """Scalar value of the critic readout, shaped (n, 1) for both heads."""
        if self.histogram is None:
            return readout
        return self.histogram.decode(readout).unsqueeze(-1)

    def get_value(self, x):
        return self.decode(self.critic(x))

    def get_policy_and_readout(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training retains native samples."""
        alpha, beta, readout = self.get_policy_and_readout(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, self.decode(readout)


def clipping_parameters(agent, clip_heads):
    """Select each independent budget once; excluded heads still belong to Adam."""
    actor = agent.actor if clip_heads else agent.actor[0]
    critic = agent.critic if clip_heads else agent.critic[0]
    return tuple(actor.parameters()), tuple(critic.parameters())


def clip_gradients(actor_parameters, critic_parameters, max_grad_norm):
    """Clip each network independently after the weighted joint loss backward."""
    actor_preclip_norm = nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm, foreach=True)
    critic_preclip_norm = nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm, foreach=True)
    return actor_preclip_norm, critic_preclip_norm


def scalar_value_loss(newvalue, targets, old_values, args):
    """PPO's clipped scalar regression, unchanged from the baseline."""
    squared = (newvalue - targets) ** 2
    if not args.clip_vloss:
        return 0.5 * squared.mean(), torch.zeros((), device=newvalue.device)
    clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
    clipped_squared = (clipped - targets) ** 2
    frozen = (clipped_squared > squared).float()
    return 0.5 * torch.max(squared, clipped_squared).mean(), frozen.mean()


def categorical_value_loss(agent, readout, newvalue, targets, old_values, target_probs, args):
    """Cross-entropy on frozen HL-Gauss labels with the decoded-value trust region.

    The trust region is the scalar rule's own decision, applied to the decoded
    value: PPO's `max(unclipped, clipped)` contributes zero value gradient
    exactly when the clipped branch is the larger error, because `clamp` is
    saturated there. The same sample set is dropped from the cross-entropy, so
    the clipping budget matches the indclip_v4 scalar baseline's.
    """
    log_probs = readout.log_softmax(dim=-1)
    cross_entropy = -(target_probs * log_probs).sum(dim=-1)
    if args.ce_sample_weight == "variance":
        with torch.no_grad():
            variance = agent.histogram.standardized_variance(log_probs.exp())
            weight = variance / variance.mean().clamp_min(1e-12)
        cross_entropy = cross_entropy * weight
    if args.clip_vloss:
        with torch.no_grad():
            clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
            frozen = ((clipped - targets).square() > (newvalue - targets).square()).float()
        cross_entropy = cross_entropy * (1.0 - frozen)
        clipfrac = frozen.mean()
    else:
        clipfrac = torch.zeros((), device=readout.device)
    if args.ce_scale == "matched":
        # The CE's value-space gradient is (V - y) / (scale^2 * Var_p(z)), so the
        # multiplier must be the CURRENT head variance, not the converged-head
        # constant sigma^2 + w^2/12. Early in training Var_p(z) is 15-570x that
        # constant, so the constant form starves the critic by the same factor
        # instead of matching 0.5*MSE. One detached scalar; graph shape unchanged.
        with torch.no_grad():
            gain = agent.histogram.scale.square() * agent.histogram.standardized_variance(
                log_probs.exp()
            ).mean()
        cross_entropy = cross_entropy * gain
    return cross_entropy.mean(), clipfrac


def policy_terms(agent, observations, native_actions, old_logprobs, advantages, args):
    alpha, beta, readout = agent.get_policy_and_readout(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return readout, pg_loss, entropy.mean(), (old_approx_kl, approx_kl, clipfrac)


def assemble(pg_loss, entropy_loss, v_loss, v_clipfrac, diagnostics, args):
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    old_approx_kl, approx_kl, clipfrac = diagnostics
    metrics = torch.stack(
        (
            pg_loss.detach(),
            v_loss.detach(),
            entropy_loss.detach(),
            old_approx_kl,
            approx_kl,
            clipfrac,
            v_clipfrac,
        )
    )
    return loss, metrics


def scalar_ppo_loss(agent, observations, native_actions, old_logprobs, advantages, targets, old_values, args):
    readout, pg_loss, entropy_loss, diagnostics = policy_terms(
        agent, observations, native_actions, old_logprobs, advantages, args
    )
    newvalue = agent.decode(readout).view(-1)
    v_loss, v_clipfrac = scalar_value_loss(newvalue, targets, old_values, args)
    return assemble(pg_loss, entropy_loss, v_loss, v_clipfrac, diagnostics, args)


def categorical_ppo_loss(
    agent, observations, native_actions, old_logprobs, advantages, targets, old_values, target_probs, args
):
    readout, pg_loss, entropy_loss, diagnostics = policy_terms(
        agent, observations, native_actions, old_logprobs, advantages, args
    )
    newvalue = agent.histogram.decode(readout)
    if args.value_loss == "mse_softmax":
        v_loss, v_clipfrac = scalar_value_loss(newvalue, targets, old_values, args)
    else:
        v_loss, v_clipfrac = categorical_value_loss(
            agent, readout, newvalue, targets, old_values, target_probs, args
        )
    return assemble(pg_loss, entropy_loss, v_loss, v_clipfrac, diagnostics, args)


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.critic_width <= 0:
        raise ValueError("critic_width must be positive")
    if args.value_loss != "hlgauss" and (args.ce_scale != "raw" or args.ce_sample_weight != "none"):
        raise ValueError("ce_scale and ce_sample_weight only apply to value_loss='hlgauss'")
    if args.logit_softcap and args.value_loss == "mse":
        raise ValueError("logit_softcap applies to the categorical readout, not the scalar head")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    return args


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
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        histogram = agent.histogram
        writer.add_text(
            "critic",
            f"value_loss={args.value_loss}; atoms={args.value_atoms}; "
            f"half_span={args.value_half_span} target std; "
            f"bin_width={0 if histogram is None else histogram.bin_width:.6g} target std; "
            f"sigma={0 if histogram is None else histogram.sigma:.6g} target std; "
            f"bins_per_target_std={0 if histogram is None else 1.0 / histogram.bin_width:.4g}; "
            f"ce_scale={args.ce_scale}; ce_sample_weight={args.ce_sample_weight}; "
            f"logit_softcap={args.logit_softcap}; "
            f"critic_width={args.critic_width}; clip_vloss={args.clip_vloss}",
        )
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        actor_parameters, critic_parameters = clipping_parameters(agent, args.clip_heads)
        clip_scope = "" if args.clip_heads else "_trunk"
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, readout = agent.get_policy_and_readout(observations)
            return agent.decode(readout).flatten(), agent.action_logprob(alpha, beta, native)

        if histogram is None:

            def loss_model(observations, native, old_logprobs, advantages, targets, old_values, target_probs):
                return scalar_ppo_loss(
                    agent, observations, native, old_logprobs, advantages, targets, old_values, args
                )

        else:

            def loss_model(observations, native, old_logprobs, advantages, targets, old_values, target_probs):
                return categorical_ppo_loss(
                    agent, observations, native, old_logprobs, advantages, targets, old_values, target_probs, args
                )

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sample_beta_actions_host(host_actor(observations), action_low, action_high, sampler)
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
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma) if args.reward_norm else None
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 7), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        placeholder = torch.zeros(
            (args.minibatch_size, args.value_atoms if histogram is not None else 1), device=device
        )
        saturated = label_error = torch.zeros((), device=device)
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
                    reward = rew_norm.normalize(raw_reward, terms) if rew_norm is not None else raw_reward
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
                b_old_values = b_values
                b_target_probs = placeholder
                if histogram is not None:
                    previous_mean, previous_scale = histogram.mean.clone(), histogram.scale.clone()
                    histogram.observe(b_returns)
                    # The head is unchanged, only the readout convention: move the
                    # rollout values into the new frame so the value trust region
                    # measures policy-update movement, not a normalizer step.
                    b_old_values = histogram.mean + histogram.scale * (b_values - previous_mean) / previous_scale
                    # Targets are constant across epochs, so the (32768, atoms)
                    # labels are built exactly once per iteration, before the
                    # epoch loop; `project` clips into the support like
                    # kraggiculture does, and the discarded fraction is logged
                    # rather than aborting the run.
                    saturated = histogram.saturated(b_returns)
                    b_target_probs = histogram.project(b_returns)
                    label_error = (histogram.decode_probs(b_target_probs) - b_returns).abs().mean()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices],
                            b_native[indices],
                            b_logprobs[indices],
                            b_advantages[indices],
                            b_returns[indices],
                            b_old_values[indices],
                            b_target_probs[indices] if histogram is not None else placeholder,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        actor_preclip_norm, critic_preclip_norm = clip_gradients(
                            actor_parameters,
                            critic_parameters,
                            args.max_grad_norm,
                        )
                        grad_norms[updates, 0].copy_(actor_preclip_norm)
                        grad_norms[updates, 1].copy_(critic_preclip_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            performed_grad_norms = grad_norms[:updates]
            mean_grad_norms = performed_grad_norms.mean(dim=0)
            grad_clip_fractions = (performed_grad_norms > args.max_grad_norm).float().mean(dim=0)
            scalars = {
                "losses/policy_loss": last[0],
                "losses/value_loss": last[1],
                "losses/entropy": last[2],
                "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4],
                "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/value_clipfrac": update_metrics[:updates, 6].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                f"grad/actor{clip_scope}_preclip_norm": mean_grad_norms[0],
                f"grad/critic{clip_scope}_preclip_norm": mean_grad_norms[1],
                f"grad/actor{clip_scope}_clip_fraction": grad_clip_fractions[0],
                f"grad/critic{clip_scope}_clip_fraction": grad_clip_fractions[1],
            }
            if histogram is not None:
                scalars.update(
                    {
                        "value/target_mean": histogram.mean,
                        "value/target_scale": histogram.scale,
                        "value/saturated_fraction": saturated,
                        "value/label_decode_error": label_error,
                    }
                )
            logged = gather_metrics(scalars)
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar(
                "charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step
            )
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(
                {
                    "model": agent.state_dict(),
                    "args": vars(args),
                    "obs_norm": {
                        "means": torch.from_numpy(obs_norm.means.copy()),
                        "variances": torch.from_numpy(obs_norm.variances.copy()),
                        "counts": torch.from_numpy(obs_norm.counts.copy()),
                        "epsilon": obs_norm.epsilon,
                        "clip": obs_norm.clip,
                    },
                },
                model_path,
            )
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
