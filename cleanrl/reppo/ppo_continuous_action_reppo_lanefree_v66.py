# REPPO v66 lanefree: v65 (v60 + read-only instruments, --rng-stream, four default-off arms) with two more
# default-off arms. At default flags every traced loss, update step, host normalization call and the update
# loop's work are v65's, so the trajectory is v60's.
#   --obs-norm shared     v60 normalizes each env lane with its own running statistics, live at every step, and
#                         the update re-normalizes the rollout with the end-of-rollout ones. Two things follow:
#                         one physical state reaches the shared networks through 16 affine maps (the lanes'
#                         policies sit a median 2-5 KL bounds apart at the same state), and old_actor on the
#                         update's inputs is not the behaviour policy. `shared` changes BOTH, and they are two
#                         ingredients, not one:
#                           pooling   one statistic for every lane: the exact moments of the union of the
#                                     lanes' samples (their own variances plus the spread of their means);
#                           freezing  that statistic is read once, when a rollout starts, and held: the host
#                                     actor and the update normalize the rollout with the same FP32 numbers,
#                                     so the KL anchor is the behaviour policy at the inputs it acted on.
#                         The lanes' accumulators keep running underneath (same samples, same order as v60).
#                         Evaluation and saved statistics are the pooled ones the next rollout would freeze;
#                         saved lane rows all hold them, so lane-indexed readers and either mode's resume work.
#                         The unrecorded phase warmup still acts on live lane statistics (nothing is frozen yet).
#   --kl-mode penalty     v60 gates per state: a state inside --kl-bound gets -Q + T log pi and no KL term, one
#                         outside gets eta KL(old || new) and nothing else (late in training the clipped actor
#                         gradient is mostly that restoring force). `penalty` is the smooth Lagrangian
#                         mean_s[-Q + T log pi + eta KL_s]: every state gets both terms. Same KL, same eta dual
#                         (it moves the batch-mean KL to --kl-bound), so the bound now holds on average rather
#                         than per state, and before eta has grown from --kl-start nothing holds it at all.
#                         policy/kl, kl_violation (share of states above the bound) and kl_multiplier compare modes.
#   --init-sigma S       quiet, state-independent start: the policy head is re-drawn orthogonal at gain 0.01
#                         (mean ~ 0) with the log-std bias at log S, from its own generator (the critic and the
#                         actor trunk keep --seed's init). Startup reports the measured action spread.
#   --aux-every N         the latent roll and reward losses only on every Nth critic step; the others run a
#                         second captured step with the categorical loss alone (no roll or reward forward). One
#                         AdamW serves both; on those steps the dynamics / reward-head parameters, which the
#                         categorical loss does not reach, are withheld from it (no momentum coasting, weight
#                         decay or second-moment decay), so they take exactly one Adam step per auxiliary
#                         gradient. --reward-latents M: reward head on z_0..z_{M-1} only.
#   --dqda-norm clip|unit conditions the pathwise signal per state. v60's actor takes one action per state and
#                         differentiates -Q(s, a) through it, i.e. applies -g da/dtheta with g = dQ/da. Here g is
#                         taken as a constant, conditioned across the minibatch's rows (clip: norms capped at
#                         their 99th percentile; unit: every norm set to the rows' RMS norm, so directions and
#                         the summed energy are kept), and enters as the surrogate -(a . g). Statistics are over
#                         all rows of the minibatch; the KL gate then drops rows exactly as in v60, and the
#                         entropy and KL terms are untouched. `none` does not route through the surrogate.
#   --actor-critic        the network the actor differentiates: live (v60), ema (weight average of the critic,
#                         stepped inside each captured critic step, horizon --actor-critic-horizon iterations,
#                         started from the critic or the checkpoint's) or frozen (the critic at the start of
#                         the iteration). Targets, bootstraps and the critic's own fit are unchanged.
# Declared deviations, as v60: fixed reward scale instead of VectorRewardNorm, strict FP32.
import copy
import math
import os
import random
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.graph_step import capture_steps
from cleanrl.shared.hl_gauss import HLGaussConfig
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))
LOG_2 = float(np.log(2.0))
ZERO_DIST_SCALE = 40.9  # REPPO's fixed multiplier on the learned zero-value logit offset
CRITIC_METRICS = ("critic_ce", "aux_loss", "reward_loss", "q_mse", "support_clip_frac", "target_mean")
ACTOR_METRICS = ("actor_objective", "q_pi", "entropy", "kl", "kl_violation", "temperature", "kl_multiplier",
                 "std", "saturation", "mean_abs_action")
DIAGNOSTIC_METRICS = ("dqda_norm", "advantage_corr_heldout", "advantage_std_ratio")
PROBE_METRICS = ("true_curvature", "true_curvature_se", "critic_curvature", "maxent_curvature",
                 "true_curvature_min_dim", "true_curvature_max_dim", "critic_curvature_min_dim",
                 "slope_cross", "slope_cross_se", "slope_energy", "slope_self_cross", "slope_beta",
                 "real_advantage_std", "critic_contrast_std")
POSTURE_METRICS = ("inverted_frac", "tumbled_env_frac")
SPLIT_METRICS = ("std_upright", "std_inverted", "q_mse_upright", "q_mse_inverted", "rows_upright", "rows_inverted")
CONDITIONING_METRICS = ("top1_energy_share", "grad_norm_median", "grad_norm_mean", "slope_cos_median",
                        "amp_ratio_p10", "amp_ratio_median", "amp_ratio_p90", "rewrite_rel", "move_cos_median",
                        "kl_gate_excluded_frac", "top1_inverted_share")
GAIN_METRICS = ("realised", "realised_se", "realised_mean_part", "realised_hard", "predicted_pre",
                "predicted_post", "predicted_pre_mean_part")
IDS_METRICS = ("target_temperature_mean", "target_temperature_min", "target_temperature_max",
               "offset_regret", "noise_regret", "unidentified_dim_frac", "entropy_slope_energy_frac")


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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """AdamW learning rate of both the actor and the critic"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    minibatch_size: int = 2048
    """rows per critic and actor update"""
    gve_lambda: float = 0.8
    """lambda of the lambda-return critic target"""
    target_refresh: int = 16
    """critic updates between recomputations of the targets from the live critic"""
    bootstrap_samples: int = 4
    """current-policy actions averaged into each bootstrap's soft value"""
    diagnostic_lambda: float = 0.95
    """lambda of the held-out diagnostic returns (fixed across --gve-lambda)"""
    critic_updates: int = 128
    """critic updates per rollout, each on a uniform minibatch of the fresh rollout"""
    actor_updates: int = 32
    """actor updates per rollout, interleaved evenly with the critic's"""
    max_grad_norm: float = 0.5
    """per-network gradient clipping norm"""
    reward_scale: float = 0.1
    """fixed reward multiplier (keeps the categorical support stationary)"""

    # Critic
    critic_hidden_dim: int = 512
    """critic width"""
    num_bins: int = 151
    """HL-Gauss categories"""
    v_min: float = -50.0
    """lowest supported scaled return"""
    v_max: float = 250.0
    """highest supported scaled return"""
    aux_loss_coef: float = 1.0
    """weight of the latent rollout loss (mean over its valid steps)"""
    aux_horizon: int = 8
    """latent rollout steps H"""
    reward_loss_coef: float = 1.0
    """weight of the latent reward prediction loss (scaled rewards, mean over z_0..z_{H-1})"""

    # Actor
    actor_hidden_dim: int = 512
    """actor width"""
    ent_start: float = 0.01
    """initial entropy temperature (kept until the critic has been fit once)"""
    temperature_shape: str = "dim"
    """dim: one information price per action dimension; scalar: one shared price (RMS over dimensions)"""
    offset_reference: str = "belief"
    """belief: previous policy under the current critic; lag: current policy under the critic it chased"""
    temperature_source: str = "dual"
    """ids: v54's information price; dual: v36's entropy-target dual (the reference regime)"""
    ent_target_mult: float = 0.5
    """dual: target entropy is -ent_target_mult * action_dim"""
    ids_samples: int = 8
    """policy samples per state in the offset-slope estimate (split into two independent halves)"""
    kl_bound: float = 0.03
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
    dual_step_per_iteration: float = 0.15
    """log-space step budget of each dual variable and of the temperature per iteration (REPPO: 512 x 3e-4)"""
    obs_clip: float = 10.0
    """observation normalization clip (VectorObsNorm's)"""

    # Diagnostics
    diagnostic_rows: int = 4096
    """fresh-rollout rows of the per-iteration critic diagnostics"""
    diagnostic_samples: int = 8
    """policy samples in the diagnostics' state baseline E_{a~pi} Q(s, a)"""
    eval_interval: int = 25
    """iterations between deterministic (mean-action) evaluation episodes; 0 disables"""
    eval_episodes: int = 8
    """mean-action episodes per evaluation (own env and seeds; the first is v60's single episode)"""
    rng_stream: int = 0
    """replicate stream of the host action-noise sampler (0: v60's); nothing else is reseeded"""
    instrument_interval: int = 4
    """iterations between the split, conditioning and gain instruments (posture is every iteration)"""
    instrument_rows: int = 4096
    """fresh-rollout rows of the gradient-conditioning instrument"""

    # Arms (all off by default)
    init_sigma: float = 0.0
    """if positive, initial pre-tanh std at every state (policy head at gain 0.01); 0 keeps v60's init"""
    aux_every: int = 1
    """critic steps per latent-roll and reward loss evaluation (the categorical loss runs on every step)"""
    reward_latents: int = 0
    """rolled latents z_0..z_{M-1} the reward head is fit on; 0 means all --aux-horizon of them"""
    dqda_norm: str = "none"
    """per-state conditioning of the actor's dQ/da: none, clip (99th percentile norm) or unit (RMS norm)"""
    actor_critic: str = "live"
    """network the actor differentiates: live, ema (weight average) or frozen (start of the iteration)"""
    actor_critic_horizon: float = 4.0
    """averaging horizon of --actor-critic ema, in iterations"""
    obs_norm: str = "lane"
    """observation statistics: lane (per env, live; v60) or shared (pooled over envs, frozen during a rollout)"""
    kl_mode: str = "gate"
    """trust region: gate (per-state switch between the objective and the KL pull-back; v60) or penalty"""
    snapshot_steps: str = ""
    checkpoint_steps: str = ""
    """comma-separated global steps at which to save a full training-state checkpoint"""
    resume: str = ""
    """full-state checkpoint to fork from; the fork then runs --iterations iterations"""
    iterations: int = 0
    """if positive, the number of iterations to run (overrides --total-timesteps)"""
    fork_ent_target_mult: float = -1.0
    """entropy-target multiplier during the fork's first --fork-iterations iterations (negative: no perturbation)"""
    fork_iterations: int = 0
    """length of the perturbation window; afterwards the target returns to --ent-target-mult"""
    """comma-separated global steps (e.g. 4e6,12e6) after which actor, critic and normalization are saved"""

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile the losses and capture both update steps as CUDA graphs"""
    matmul_precision: str = "highest"
    """torch float32 matmul precision: highest (strict FP32, v27), high (TF32)"""
    bf16: bool = False
    """BF16 autocast for the critic and actor update losses (reductions and log-softmax stay FP32)"""
    profile_iteration: int = 0
    """iteration whose update phase is profiled into runs/<run>/update_profile.txt; 0 disables"""
    compile_mode: str = "reduce-overhead"
    """accepted for launcher compatibility; every path here uses Inductor without cudagraph trees"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def normed_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), nn.RMSNorm(out_dim, eps=1e-5), nn.SiLU()]


def policy_moments(net, observations):
    """Pre-squash mean and standard deviation."""
    mean, log_std = net(observations).chunk(2, dim=-1)
    return mean, log_std.exp()


def tanh_log_det(pre):
    """log|d tanh(u)/du| = 2 (log 2 - u - softplus(-2u)), stable for large |u|."""
    return 2.0 * (LOG_2 - pre - F.softplus(-2.0 * pre))


def squashed_sample(mean, std):
    """Reparameterized tanh-Gaussian sample and its exact per-dimension log-density (the policy factorizes)."""
    noise = torch.randn_like(mean)
    pre = mean + std * noise
    logprob = -0.5 * noise.square() - std.log() - LOG_SQRT_2PI - tanh_log_det(pre)
    return torch.tanh(pre), logprob


def gaussian_kl(old_mean, old_std, mean, std):
    """KL(old || new) of diagonal Gaussians, equal to the KL of their tanh pushforwards."""
    return (torch.log(std / old_std)
            + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square()) - 0.5).sum(-1)


def action_noise_rng(seed, stream):
    """Host action-noise generator. Stream 0 is default_rng(seed): SeedSequence zero-pads its entropy (an
    implementation detail of NumPy, pinned by tests/test_reppo_instrument_v64.py)."""
    return np.random.default_rng([seed, stream])


def inverted_rows(pitch):
    """On its back, from the unwrapped root pitch: a completed somersault (2 pi) is upright again."""
    return torch.cos(pitch) < 0.0


def tumbled_envs(inverted, next_inverted):
    """Per env (columns of (steps, envs) masks): some step started upright and ended inverted."""
    return (~inverted & next_inverted).any(0)


def top_energy_share(energy, fraction=0.01):
    """Share of sum(energy) held by the top `fraction` of rows, and those rows."""
    top = energy.topk(max(1, round(fraction * energy.numel())))
    return top.values.sum() / energy.sum().clamp_min(1e-30), top.indices


def row_cosine(first, second):
    return F.cosine_similarity(first, second, dim=-1, eps=1e-12)


def gaussian_log_ratio(noise, mean, std, old_mean, old_std):
    """log pi(u) - log pi_old(u) at u = mean + std * noise, diagonal Gaussians (the tanh Jacobian cancels)."""
    old_noise = (mean + std * noise - old_mean) / old_std
    return (0.5 * (old_noise.square() - noise.square()) + torch.log(old_std / std)).sum(-1)


def centered_log_ratio(noise, mean, std, old_mean, old_std):
    """gaussian_log_ratio minus its expectation over the action, KL(pi || pi_old) at that state. E_u[A w] is
    unchanged for a true advantage (E_u[A] = 0), but an estimated one, G - b(s), no longer leaks its
    per-state baseline error b - V through the (positive, state-dependent) KL."""
    return gaussian_log_ratio(noise, mean, std, old_mean, old_std) - gaussian_kl(mean, std, old_mean, old_std)


def realised_gain_terms(advantages, log_ratio):
    """Rows of the backward-form gain: with (s, u) ~ pi and A the advantage of pi, the mean of
    (A - mean A) [log pi(u|s) - log pi_old(u|s)] is (1 - gamma) (J(pi) - J(pi_old)) to first order.
    Centering A keeps a nonzero sample mean of the log ratio from leaking the return's level in; pass the
    per-state centered log ratio to keep state-dependent baseline error out as well."""
    return (advantages - advantages.mean()) * log_ratio


def lane_standard_error(terms, num_envs):
    """Standard error of terms.mean() from the envs' own means (rows in (step, env) order): with equal lanes
    this is the lane jackknife, which respects each lane's serial dependence."""
    return terms.view(-1, num_envs).mean(0).std() / num_envs ** 0.5


def ema_decay(horizon_iterations, steps_per_iteration):
    """Per-step decay of an average whose horizon 1 / (1 - decay) spans the given number of iterations."""
    return 1.0 - 1.0 / (horizon_iterations * steps_per_iteration)


def upper_order_statistic(values, quantile):
    """Smallest entry with at least `quantile` of the entries at or below it. By pairwise counting: only
    elementwise kernels, so it can sit in a captured CUDA graph without a sort."""
    need = math.ceil(quantile * values.numel())
    at_or_below = (values[None, :] <= values[:, None]).sum(1)
    return values.masked_fill(at_or_below < need, torch.inf).min()


def clip_slope(slope, quantile=0.99):
    """Rows' norms capped at their `quantile` order statistic; directions kept."""
    norm = slope.norm(dim=-1, keepdim=True)
    limit = upper_order_statistic(norm.squeeze(-1), quantile)
    return slope * (limit / norm.clamp_min(1e-12)).clamp_max(1.0)


def unit_slope(slope):
    """Every row at the rows' RMS norm: directions and the summed energy kept, amplitudes equalized."""
    norm = slope.norm(dim=-1, keepdim=True)
    return slope * (norm.square().mean().sqrt() / norm.clamp_min(1e-12))


SLOPE_CONDITIONERS = {"clip": clip_slope, "unit": unit_slope}


def gated_policy_loss(objective, kl, multiplier, bound):
    """v60's trust region: a state inside the bound receives its objective, one outside only the pull back."""
    return torch.where(kl < bound, objective, kl * multiplier).mean()


def penalized_policy_loss(objective, kl, multiplier, bound):
    """The Lagrangian of mean KL <= bound at a fixed multiplier: every state receives both terms. The bound
    only enters through the multiplier's own update."""
    return (objective + kl * multiplier).mean()


KL_POLICY_LOSSES = {"gate": gated_policy_loss, "penalty": penalized_policy_loss}


def pooled_moments(means, variances, counts):
    """Mean, variance and count of the union of the lanes' samples, from each lane's own (rows of `means` and
    `variances`, population moments over `counts` samples). The union's variance is the count-weighted mean of
    the lanes' variances plus that of their means' squared offsets from the pooled mean."""
    total = counts.sum()
    weights = counts / total
    mean = np.tensordot(weights, means, axes=1)
    variance = np.tensordot(weights, variances + np.square(means - mean), axes=1)
    return mean, variance, total


def frozen_normalize(raw, mean, inv_std, clip):
    """FP32 (raw - mean) * inv_std, clipped: the arithmetic the update applies to stored raw observations, so
    host and GPU inputs agree to the bit. A new array on every call (rollout rows keep references)."""
    out = np.subtract(raw, mean, dtype=np.float32)
    np.multiply(out, inv_std, out=out)
    return np.clip(out, -clip, clip, out=out)


class SharedObsNorm:
    """--obs-norm shared: one statistic for all lanes, pooled from a per-lane accumulator (a VectorObsNorm,
    which keeps counting every observation in wrapper order) and held between calls to freeze()."""

    def __init__(self, accumulator):
        self.accumulator = accumulator
        self.mean = self.inv_std = None  # FP32, valid from the first freeze()

    def pooled(self):
        """The accumulator's pooled moments as they stand now (float64)."""
        accumulator = self.accumulator
        return pooled_moments(accumulator.means, accumulator.variances, accumulator.counts)

    def scale(self):
        """FP32 mean and reciprocal standard deviation of the pooled moments as they stand now."""
        mean, variance, _ = self.pooled()
        return mean.astype(np.float32), (1.0 / np.sqrt(variance + self.accumulator.epsilon)).astype(np.float32)

    def freeze(self):
        """Adopt the current pooled statistics. New arrays: earlier references keep the previous freeze."""
        self.mean, self.inv_std = self.scale()

    def normalize(self, raw):
        return frozen_normalize(raw, self.mean, self.inv_std, self.accumulator.clip)

    def lane_state(self):
        """(means, variances, counts) in the accumulator's per-lane layout, every lane holding the pooled
        moments and an equal share of the count: equal lanes pool back to the same moments (to rounding), so
        lane-indexed readers see the shared statistic and a resumed accumulator loses nothing."""
        accumulator = self.accumulator
        mean, variance, total = self.pooled()
        lanes = len(accumulator.counts)
        return (np.broadcast_to(mean, accumulator.means.shape).copy(),
                np.broadcast_to(variance, accumulator.variances.shape).copy(), np.full(lanes, total / lanes))


@contextmanager
def withheld_gradients(parameters):
    """Hide the parameters' gradients from an optimizer step taken inside: torch optimizers skip a parameter
    whose .grad is None entirely (no update, no moment decay, no weight decay, no step count), where a zero
    gradient would still coast on momentum and shrink the second moment. The same gradient tensors are put
    back, so graphs that captured their storage keep writing into it."""
    gradients = [parameter.grad for parameter in parameters]
    for parameter in parameters:
        parameter.grad = None
    try:
        yield
    finally:
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient


def action_slope(model, observations, actions):
    """dQ/da and Q at the given actions, both as constants."""
    def total(candidate):
        q = model.value(observations, candidate)
        return q.sum(), q

    slope, q = torch.func.grad(total, has_aux=True)(actions)
    return slope.detach(), q.detach()


def init_uniform_sigma(head, sigma, seed, gain=0.01):
    """State-independent start of a (mean, log_std) head whose std is exp(log_std), unbounded: orthogonal
    weights at a small gain leave mean ~ 0 and log_std ~ its bias, set to log(sigma). Drawn on the CPU from
    its own generator, so every other module keeps the run's init."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.weight.copy_(nn.init.orthogonal_(torch.empty(head.weight.shape), gain=gain, generator=generator))
        mean_bias, log_std_bias = head.bias.chunk(2)
        mean_bias.zero_()
        log_std_bias.fill_(float(np.log(sigma)))


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, args):
        super().__init__()
        width = args.actor_hidden_dim
        # Flat stack so the fused host mirror can evaluate it in one native call.
        self.net = nn.Sequential(*normed_layer(observation_dim, width), *normed_layer(width, width),
                                 nn.Linear(width, 2 * action_dim))
        # Per-dimension information price, set from measurements (ids); or v36's scalar dual (dual).
        self.register_buffer("log_temperature", torch.full((action_dim,), float(np.log(args.ent_start))))
        self.log_dual_temperature = nn.Parameter(torch.tensor(float(np.log(args.ent_start))))
        self.dual = args.temperature_source == "dual"
        self.log_kl_multiplier = nn.Parameter(torch.tensor(float(np.log(args.kl_start))))

    def forward(self, observations):
        return policy_moments(self.net, observations)

    def temperature(self):
        if self.dual:
            return self.log_dual_temperature.exp().expand_as(self.log_temperature)
        return self.log_temperature.exp()


class Critic(nn.Module):
    """REPPO's Q(s, a): encoder -> categorical value head, plus a latent transition and reward head."""

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        width = args.critic_hidden_dim
        self.encoder = nn.Sequential(*normed_layer(observation_dim + action_dim, width), nn.Linear(width, width))
        self.value_head = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, args.num_bins))
        # Latent transition g(z, a') and reward head r_hat(z).
        self.dynamics = nn.Sequential(*normed_layer(width + action_dim, width), nn.Linear(width, width))
        self.reward_head = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, 1))
        self.hl = HLGaussConfig(v_min=args.v_min, v_max=args.v_max, num_bins=args.num_bins).build(device)
        self.zero_dist = nn.Parameter(self.hl.project(torch.zeros((), device=device)))

    def features(self, observations, actions):
        return self.encoder(torch.cat((observations, actions), dim=-1))

    def transition(self, features, actions):
        return self.dynamics(torch.cat((F.silu(features), actions), dim=-1))

    def reward(self, features):
        return self.reward_head(features).squeeze(-1)

    def logits(self, features):
        return self.value_head(features) + ZERO_DIST_SCALE * self.zero_dist

    def value(self, observations, actions):
        return self.hl.to_scalar(self.logits(self.features(observations, actions)))


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs."""

    def __init__(self, num_envs, action_dim):
        shape = (num_envs, action_dim)
        self.std = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)
        self.noise = np.empty(shape, dtype=np.float32)
        self.mean = None  # view of the last host output's mean half (valid until the next host forward)

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
        self.mean = mean
        np.exp(log_std, out=self.std)
        rng.standard_normal(self.noise.shape, dtype=np.float32, out=self.noise)
        np.multiply(self.noise, self.std, out=self.action)
        self.action += mean
        np.tanh(self.action, out=self.action)
        return self.action


def lambda_returns(gae_fn, rewards, next_values, terminations, truncations, gamma, lam):
    """REPPO's GVE target G_t = r_t + gamma [(1 - lam) Q'_t + lam G_{t+1}] via the shared GAE kernel.

    Q'_t = Q(s_{t+1}, a'), a' ~ pi. Boundaries bootstrap from Q'_t alone; terminations do not
    bootstrap. Explicit-next-value GAE returns A_t + v_t with
    A_t = r_t + gamma Q'_t - v_t + gamma lam A_{t+1}, which is exactly G_t when v_{t+1} = Q'_t:
    set v_t = Q'_{t-1} (v_0 cancels).
    """
    values = torch.cat((torch.zeros_like(next_values[:1]), next_values[:-1]))
    _, returns = gae_fn(rewards, values, terminations, truncations, next_values, gamma, lam)
    return returns


def validate_args(args):
    if args.temperature_shape not in ("dim", "scalar"):
        raise ValueError("--temperature-shape must be dim or scalar")
    if args.temperature_source not in ("ids", "dual"):
        raise ValueError("--temperature-source must be ids or dual")
    if args.offset_reference not in ("belief", "lag"):
        raise ValueError("--offset-reference must be belief or lag")
    if args.ids_samples < 2 or args.ids_samples % 2:
        raise ValueError("--ids-samples must be even and at least 2 (two independent halves)")
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates) <= 0:
        raise ValueError("environment, rollout, minibatch and update counts must be positive")
    if args.aux_horizon <= 0 or args.aux_horizon >= args.num_steps:
        raise ValueError("aux_horizon must be in [1, num_steps)")
    if args.target_refresh <= 0 or args.critic_updates % args.target_refresh or args.bootstrap_samples <= 0:
        raise ValueError("target_refresh must divide critic_updates; bootstrap_samples positive")
    if args.ent_start <= 0.0:
        raise ValueError("the initial entropy temperature must be positive")
    args.snapshot_at = sorted(int(float(step)) for step in args.snapshot_steps.split(",") if step.strip())
    args.checkpoint_at = sorted(int(float(step)) for step in args.checkpoint_steps.split(",") if step.strip())
    if args.fork_ent_target_mult >= 0 and not (args.resume and 0 < args.fork_iterations):
        raise ValueError("--fork-ent-target-mult needs --resume and a positive --fork-iterations")
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    if not 0 < args.diagnostic_rows <= args.batch_size or args.diagnostic_samples < 1 or args.eval_interval < 0:
        raise ValueError("diagnostic rows must fit the rollout; samples positive; eval interval nonnegative")
    if args.rng_stream < 0 or args.eval_episodes < 1 or args.instrument_interval < 1:
        raise ValueError("rng stream nonnegative; eval episodes and instrument interval positive")
    if not 0 < args.instrument_rows <= args.batch_size:
        raise ValueError("instrument rows must fit the rollout")
    if args.init_sigma < 0.0 or (args.init_sigma and args.resume):
        raise ValueError("--init-sigma must be nonnegative and only applies to a run from scratch")
    if args.aux_every < 1 or not 0 <= args.reward_latents <= args.aux_horizon:
        raise ValueError("--aux-every must be positive and --reward-latents within [0, aux_horizon]")
    if args.dqda_norm not in ("none", *SLOPE_CONDITIONERS):
        raise ValueError("--dqda-norm must be none, clip or unit")
    if args.actor_critic not in ("live", "ema", "frozen"):
        raise ValueError("--actor-critic must be live, ema or frozen")
    if args.actor_critic_horizon * args.critic_updates < 1.0:
        raise ValueError("--actor-critic-horizon must span at least one critic step")
    if args.obs_norm not in ("lane", "shared"):
        raise ValueError("--obs-norm must be lane or shared")
    if args.obs_norm == "shared" and not (args.resume or (args.staggered_starts and args.num_envs > 1)):
        # Frozen statistics need samples before the first rollout: sixteen reset observations are not a scale.
        raise ValueError("--obs-norm shared needs the phase warmup or a checkpoint to seed its statistics")
    if args.kl_mode not in KL_POLICY_LOSSES:
        raise ValueError("--kl-mode must be gate or penalty")
    if not args.cuda:
        raise ValueError("REPPO requires CUDA")
    if not args.v_min < 0.0 < args.v_max:
        raise ValueError("the value support must contain zero (the critic's initial prediction)")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video, run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision=args.matmul_precision, allow_tf32=args.matmul_precision != "highest")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = args.iterations or (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        space = envs.single_action_space
        if not (isinstance(space, gym.spaces.Box) and np.all(space.low == -1.0) and np.all(space.high == 1.0)):
            raise ValueError("the tanh policy requires a [-1, 1] Box action space")
        obs_shape = envs.single_observation_space.shape
        observation_dim, action_dim = int(np.prod(obs_shape)), int(np.prod(space.shape))
        actor = Actor(observation_dim, action_dim, args).to(device)
        critic = Critic(observation_dim, action_dim, args, device).to(device)
        if args.init_sigma > 0.0:
            init_uniform_sigma(actor.net[-1], args.init_sigma, args.seed)
        # Frozen per-iteration copies: the bootstrap target and the KL anchor.
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        old_actor = copy.deepcopy(actor.net).requires_grad_(False)
        # The policy one iteration older than old_actor: the one that chased the critic before the current one.
        previous_actor = copy.deepcopy(actor.net).requires_grad_(False)
        offset_actor = previous_actor if args.offset_reference == "belief" else actor.net
        actor_network, critic_parameters = list(actor.net.parameters()), list(critic.parameters())
        dual_parameters = [actor.log_kl_multiplier] + ([actor.log_dual_temperature] if actor.dual else [])
        # A device tensor: fork arms change it between iterations and the captured graphs read its storage.
        target_entropy = torch.tensor(-args.ent_target_mult * action_dim, device=device)
        actor_optimizer = torch.optim.AdamW(
            [{"params": actor_network},
             {"params": dual_parameters, "lr": args.dual_step_per_iteration / args.actor_updates,
              "weight_decay": 0.0}],
            lr=args.learning_rate, fused=True)
        critic_optimizer = torch.optim.AdamW(critic_parameters, lr=args.learning_rate, fused=True)
        actor_parameters = actor_network + dual_parameters
        B, M, N = args.batch_size, args.minibatch_size, args.num_envs

        # GPU copy of the fresh rollout (raw observations, rows in the rollout's (step, env) order, so a
        # row's environment is row % num_envs).
        f32 = dict(device=device, dtype=torch.float32)
        rollout_obs = torch.zeros((B, observation_dim), **f32)
        rollout_next_obs = torch.zeros((B, observation_dim), **f32)
        rollout_actions = torch.zeros((B, action_dim), **f32)
        rollout_rewards = torch.zeros(B, **f32)
        rollout_continuing = torch.zeros(B, **f32)
        rollout_truncated = torch.zeros(B, **f32)
        rollout_targets = torch.zeros(B, **f32)
        # f'(s, a) of every rollout row under the current target critic (recomputed at each refresh).
        target_features = torch.zeros((B, args.critic_hidden_dim), **f32)
        # rollout_chain[k - 1, row]: row's episode continues k steps within the rollout, so row + k N holds
        # s_{t+k}, a_{t+k}, r_{t+k}.
        H = args.aux_horizon
        rollout_chain = torch.zeros((H, B), **f32)
        diagnostic_targets = torch.zeros(B, **f32)
        hard_targets = torch.zeros(B, **f32)
        rollout_noise = torch.zeros((B, action_dim), **f32)
        time_shape = (args.num_steps, N)
        rollout_ends = torch.zeros(time_shape, **f32)
        rollout_ends[-1] = 1.0
        rollout_rows = torch.arange(B, device=device)
        obs_mean = torch.zeros((N, observation_dim), **f32)
        obs_inv_std = torch.ones((N, observation_dim), **f32)
        critic_sums = torch.zeros(len(CRITIC_METRICS), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(2, **f32)
        diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed)
        # Instruments: their own generator, and the critic as it stood when the update began (target_critic is
        # refreshed inside the update). A deep copy draws nothing from the init stream.
        instrument_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        pre_critic = copy.deepcopy(critic).requires_grad_(False)
        pre_critic_iteration = None  # the iteration whose pre-update critic pre_critic holds
        posture = args.env_id.startswith("HalfCheetah")  # raw observation 1 is the unwrapped root pitch
        ratio_quantiles = torch.tensor((0.1, 0.5, 0.9), **f32)
        # Arms. The network the actor differentiates: the live critic, or a copy that either averages it
        # (stepped inside the captured critic steps) or is reset to it when each iteration starts.
        actor_critic = critic if args.actor_critic == "live" else copy.deepcopy(critic).requires_grad_(False)
        ema_parameters = list(actor_critic.parameters()) if args.actor_critic == "ema" else []
        ema_weight = 1.0 - ema_decay(args.actor_critic_horizon, args.critic_updates)
        condition_slope = SLOPE_CONDITIONERS.get(args.dqda_norm)
        trust_region_loss = KL_POLICY_LOSSES[args.kl_mode]
        reward_latents = args.reward_latents or args.aux_horizon
        # Parameters only the auxiliary losses reach: the categorical-only step leaves them alone.
        aux_only_parameters = [*critic.dynamics.parameters(), *critic.reward_head.parameters()]
        # Steps that evaluate the auxiliary losses: their logged means divide by this.
        aux_steps = len(range(0, args.critic_updates, args.aux_every))
        critic_metric_steps = [aux_steps if name in ("aux_loss", "reward_loss") else args.critic_updates
                               for name in CRITIC_METRICS]
        if not args.resume:
            # What the initial policy does, on standard-normal stand-ins for normalized observations.
            with torch.no_grad():
                probe_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
                probe_mean, probe_std = actor(
                    torch.randn((4096, observation_dim), device=device, generator=probe_generator))
                probe_actions = torch.tanh(probe_mean + probe_std * torch.randn(
                    probe_mean.shape, device=device, generator=probe_generator))
                init_report = (f"pre-tanh std mean={probe_std.mean():.4f} min={probe_std.min():.4f} "
                               f"max={probe_std.max():.4f}; action std={probe_actions.std():.4f}; "
                               f"frac(|a| > 0.9)={(probe_actions.abs() > 0.9).float().mean():.4f}")
            print(f"initial policy: {init_report}")
            writer.add_text("initial_policy", init_report)

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def observations(rows):
            return normalize(rollout_obs[rows], rows)

        def next_observations(rows):
            return normalize(rollout_next_obs[rows], rows)

        def bootstrap_values(rows):
            """Expected soft value V'(s') = mean_k [Q_frozen(s', a'_k) - T log pi(a'_k|s')] over K current-policy
            actions, and the entropy bonus T H^(s') = -T mean_k log pi(a'_k|s')."""
            next_obs = next_observations(rows)
            mean, std = actor(next_obs)
            samples = args.bootstrap_samples
            next_actions, next_logprob = squashed_sample(mean.expand(samples, -1, -1), std.expand(samples, -1, -1))
            q = target_critic.value(next_obs.expand(samples, -1, -1), next_actions)
            bonus = -(actor.temperature() * next_logprob.mean(0)).sum(-1)
            return q.mean(0) + bonus, bonus

        def critic_loss(rows):
            obs, actions, targets = observations(rows), rollout_actions[rows], rollout_targets[rows]
            # Successor rows t + k (k = 1..H), clamped in range; invalid ones are masked by the chain.
            later = torch.minimum(rows + N * torch.arange(1, H + 1, device=device)[:, None], rollout_rows[-1])
            chain = rollout_chain[:, rows]
            later_actions = rollout_actions[later]
            with torch.no_grad():
                later_features = target_features[later]
                labels = critic.hl.project(targets)
            features = critic.features(obs, actions)
            logits = critic.logits(features)
            ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            latents = [features]
            for k in range(H):
                latents.append(critic.transition(latents[-1], later_actions[k]))
            latents = torch.stack(latents)
            prediction_error = (latents[1:] - later_features).square().mean(-1)
            aux = (chain * prediction_error).sum() / chain.sum().clamp_min(1.0)
            # r_{t+k} from z_k, k = 0..H-1: z_0 always has its reward; z_k needs the chain to reach t + k.
            reward_mask = torch.cat((torch.ones_like(chain[:1]), chain[:-1]))
            reward_rows = torch.cat((rows[None], later[:-1]))
            reward_error = (critic.reward(latents[:-1]) - rollout_rewards[reward_rows]).square()
            reward_loss = (reward_mask * reward_error).sum() / reward_mask.sum()
            q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            return ce + args.aux_loss_coef * aux + args.reward_loss_coef * reward_loss, torch.stack((
                ce.detach(), aux.detach(), reward_loss.detach(), q_mse, support_clip, targets.mean()))

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs)
            mean, std = actor(obs)
            actions, logprob = squashed_sample(mean, std)
            q = critic.value(obs, actions)
            temperature = actor.temperature()
            kl_multiplier = actor.log_kl_multiplier.exp()
            objective = -q + (temperature.detach() * logprob).sum(-1)
            kl = gaussian_kl(old_mean, old_std, mean, std)
            inside = kl < args.kl_bound
            # Outside the trust region a state only receives the pull back toward the old policy.
            policy_loss = torch.where(inside, objective, kl * kl_multiplier.detach()).mean()
            entropy = -logprob.sum(-1)
            temperature_loss = (actor.log_dual_temperature.exp() * (entropy - target_entropy).detach().mean()
                                if actor.dual else torch.zeros_like(kl_multiplier))
            multiplier_loss = -kl_multiplier * (kl.mean() - args.kl_bound).detach()
            metrics = torch.stack((objective.mean(), q.mean(), entropy.mean(), kl.mean(),
                                   1.0 - inside.float().mean(), temperature.mean(), kl_multiplier, std.mean(),
                                   (actions.abs() > 0.99).float().mean(), torch.tanh(mean).abs().mean())).detach()
            return policy_loss + temperature_loss + multiplier_loss, metrics

        def reward_latent_critic_loss(rows):
            """critic_loss with the reward head fit on z_0..z_{M-1} only (--reward-latents M)."""
            obs, actions, targets = observations(rows), rollout_actions[rows], rollout_targets[rows]
            later = torch.minimum(rows + N * torch.arange(1, H + 1, device=device)[:, None], rollout_rows[-1])
            chain = rollout_chain[:, rows]
            later_actions = rollout_actions[later]
            with torch.no_grad():
                later_features = target_features[later]
                labels = critic.hl.project(targets)
            features = critic.features(obs, actions)
            logits = critic.logits(features)
            ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            latents = [features]
            for k in range(H):
                latents.append(critic.transition(latents[-1], later_actions[k]))
            latents = torch.stack(latents)
            prediction_error = (latents[1:] - later_features).square().mean(-1)
            aux = (chain * prediction_error).sum() / chain.sum().clamp_min(1.0)
            reward_mask = torch.cat((torch.ones_like(chain[:1]), chain[: reward_latents - 1]))
            reward_rows = torch.cat((rows[None], later[: reward_latents - 1]))
            reward_error = (critic.reward(latents[:reward_latents]) - rollout_rewards[reward_rows]).square()
            reward_loss = (reward_mask * reward_error).sum() / reward_mask.sum()
            q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            return ce + args.aux_loss_coef * aux + args.reward_loss_coef * reward_loss, torch.stack((
                ce.detach(), aux.detach(), reward_loss.detach(), q_mse, support_clip, targets.mean()))

        def categorical_critic_loss(rows):
            """critic_loss without the latent roll and the reward head (--aux-every's other steps)."""
            obs, actions, targets = observations(rows), rollout_actions[rows], rollout_targets[rows]
            with torch.no_grad():
                labels = critic.hl.project(targets)
            logits = critic.logits(critic.features(obs, actions))
            ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            skipped = torch.zeros_like(q_mse)
            return ce, torch.stack((ce.detach(), skipped, skipped, q_mse, support_clip, targets.mean()))

        def arm_actor_loss(rows):
            """actor_loss through --actor-critic's network, its pathwise term conditioned by --dqda-norm: the
            sampled action's slope g = dQ/da is a constant, conditioned across the minibatch's rows, in the
            first-order surrogate -(a . g), whose gradient is -g da/dtheta (the pathwise gradient of -Q when g
            is left alone). --kl-mode then combines the rows' objective and KL: the gate selects rows as in
            actor_loss; the penalty adds eta KL to every row. The multiplier's own loss is the same in both."""
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs)
            mean, std = actor(obs)
            actions, logprob = squashed_sample(mean, std)
            if condition_slope is None:
                q = actor_critic.value(obs, actions)
                pathwise = -q
            else:
                slope, q = action_slope(actor_critic, obs, actions.detach())
                pathwise = -(actions * condition_slope(slope)).sum(-1)
            temperature = actor.temperature()
            kl_multiplier = actor.log_kl_multiplier.exp()
            entropy_cost = (temperature.detach() * logprob).sum(-1)
            objective = pathwise + entropy_cost
            kl = gaussian_kl(old_mean, old_std, mean, std)
            inside = kl < args.kl_bound
            policy_loss = trust_region_loss(objective, kl, kl_multiplier.detach(), args.kl_bound)
            entropy = -logprob.sum(-1)
            temperature_loss = (actor.log_dual_temperature.exp() * (entropy - target_entropy).detach().mean()
                                if actor.dual else torch.zeros_like(kl_multiplier))
            multiplier_loss = -kl_multiplier * (kl.mean() - args.kl_bound).detach()
            # The logged objective stays v60's -Q + T log pi, whichever pathwise term carries the gradient.
            metrics = torch.stack(((entropy_cost - q).mean(), q.mean(), entropy.mean(), kl.mean(),
                                   1.0 - inside.float().mean(), temperature.mean(), kl_multiplier, std.mean(),
                                   (actions.abs() > 0.99).float().mean(), torch.tanh(mean).abs().mean())).detach()
            return policy_loss + temperature_loss + multiplier_loss, metrics

        def action_gradient(model, obs, actions):
            actions = actions.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(model.value(obs, actions).sum(), actions)
            return grad

        def critic_diagnostics(rows):
            """Is the critic's action dependence signal? Measured on fresh (held-out) rollout rows.

            advantage_corr  correlation of Q(s, a) - E_pi Q(s, .) with G_t - E_pi Q(s, .) on actions the critic
                            has not been trained on.
            """
            obs = observations(rows)
            mean, std = actor(obs)
            with torch.enable_grad():
                slope = action_gradient(target_critic, obs, torch.tanh(mean))
            samples = args.diagnostic_samples
            noise = torch.randn((samples,) + mean.shape, device=device, generator=diagnostic_generator)
            sampled = torch.tanh(mean + std * noise)
            baseline = target_critic.value(obs.expand(samples, -1, -1), sampled).mean(0)
            critic_advantage = target_critic.value(obs, rollout_actions[rows]) - baseline
            return_advantage = diagnostic_targets[rows] - baseline
            centered_c = critic_advantage - critic_advantage.mean()
            centered_r = return_advantage - return_advantage.mean()
            correlation = (centered_c * centered_r).mean() / (centered_c.std() * centered_r.std() + 1e-12)
            return torch.stack((slope.norm(dim=-1).mean(), correlation,
                                centered_c.std() / (centered_r.std() + 1e-12)))

        def information_price(rows):
            """Per-dimension IDS temperature T_i = sqrt(E_s[(sigma_i g_i)^2]) on fresh rollout states.

            g is the slope of the mean's max-ent objective, E_eps[Q(s, tanh(u)) + sum_j T_j log|dtanh(u_j)/du_j|]
            with u = mu + sigma eps (the entropy's only mean-dependent part), taken before this iteration's
            update. Two independent half-sample means give E[g1 g2] = g^2 without Monte-Carlo inflation. Also
            returns both regrets of the local model in scaled-return units: the mean's offset
            sum_i E[(sigma_i g_i)^2] / 2 T_i and the noise's sum_i T_i / 2 (equal at the IDS balance).
            """
            obs = observations(rows)
            mean, std = policy_moments(offset_actor, obs)
            temperature = actor.temperature()
            samples = args.ids_samples
            noise = torch.randn((samples,) + mean.shape, device=device, generator=diagnostic_generator)
            with torch.enable_grad():
                pre = (mean + std * noise).requires_grad_(True)
                # target_critic equals the critic here and has no parameter gradients to build.
                (q_slope,) = torch.autograd.grad(
                    target_critic.value(obs.expand(samples, -1, -1), torch.tanh(pre)).sum(), pre)
            entropy_slope = -2.0 * temperature * torch.tanh(pre)  # d/du of T log|dtanh/du|
            slope = q_slope + entropy_slope
            half = samples // 2

            def half_energy(g):
                return (std.square() * g[:half].mean(0) * g[half:].mean(0)).mean(0)

            energy = half_energy(slope)
            if args.temperature_shape == "scalar":
                energy = energy.mean().expand_as(energy)
            # A nonpositive estimate means no identified slope in that dimension: its price goes to the floor.
            target = energy.clamp_min(1e-12).sqrt()
            metrics = torch.stack((target.mean(), target.min(), target.max(),
                                   (energy / (2.0 * temperature)).sum(), temperature.sum() / 2.0,
                                   (energy <= 0.0).float().mean(),
                                   half_energy(entropy_slope).sum() / energy.sum().abs().clamp_min(1e-12)))
            return target, metrics

        def regret_probe():
            """Stein moments of the true (hard-return) and the critic's action surface at the behavior policy, over
            the whole fresh rollout; per action dimension, then summed (see header). Units: scaled return."""
            obs, noise, actions = observations(rollout_rows), rollout_noise, rollout_actions
            mean, std = actor(obs)  # the behavior policy: no update yet this iteration
            samples = args.diagnostic_samples
            draws = torch.randn((samples,) + mean.shape, device=device, generator=diagnostic_generator)
            pre = mean + std * draws
            baseline = target_critic.value(obs.expand(samples, -1, -1), torch.tanh(pre)).mean(0)
            real = hard_targets - baseline
            contrast = target_critic.value(obs, actions) - baseline
            with torch.enable_grad():
                pre = pre.requires_grad_(True)
                (slope,) = torch.autograd.grad(
                    target_critic.value(obs.expand(samples, -1, -1), torch.tanh(pre)).sum(), pre)
            half = samples // 2
            first, second = slope[:half].mean(0), slope[half:].mean(0)
            weight = noise.square() - 1.0
            true_terms = weight * real[:, None]
            true_curv = true_terms.mean(0)
            true_se = (true_terms.var(0).sum() / B).sqrt()  # of the dimension sum
            critic_curv = (weight * contrast[:, None]).mean(0)
            scaled = std * slope.mean(0)
            cross_terms = scaled * noise * real[:, None]
            cross = cross_terms.mean(0)
            cross_se = (cross_terms.sum(-1).var() / B).sqrt()
            energy = (std.square() * first * second).mean(0)
            self_cross = (scaled * noise * contrast[:, None]).mean(0)
            return torch.stack((true_curv.sum(), true_se, critic_curv.sum(), actor.temperature().sum().neg(),
                                true_curv.min(), true_curv.max(), critic_curv.min(),
                                cross.sum(), cross_se, energy.sum(), self_cross.sum(),
                                cross.sum() / energy.sum().abs().clamp_min(1e-12),
                                real.std(), contrast.std()))

        def posture_instrument():
            """Inverted rows, and the shares of rows inverted and of envs that tipped over in this rollout."""
            inverted = inverted_rows(rollout_obs[:, 1])
            tipped = tumbled_envs(inverted.view(time_shape), inverted_rows(rollout_next_obs[:, 1]).view(time_shape))
            return inverted, torch.stack((inverted.float().mean(), tipped.float().mean()))

        def split_instrument(inverted, behaviour_std):
            """Behaviour std and the updated critic's squared error against the training target (q_mse's
            definition, here over the whole rollout once the update is done), on upright and inverted rows."""
            std = behaviour_std.mean(-1)
            error = (critic.value(observations(rollout_rows), rollout_actions) - rollout_targets).square()
            upright = ~inverted
            return torch.stack((std[upright].mean(), std[inverted].mean(), error[upright].mean(),
                                error[inverted].mean(), upright.sum().float(), inverted.sum().float()))

        def conditioning_instrument(rows, inverted):
            """Per-state action slope g = dQ/da at the rollout's actions, updated critic against pre_critic, and
            the realised move of the squashed mean (the critic's action coordinates) against the pre-update g."""
            obs, actions = observations(rows), rollout_actions[rows]
            with torch.enable_grad():
                slope = action_gradient(critic, obs, actions)
                pre_slope = action_gradient(pre_critic, obs, actions)
            norm, pre_norm = slope.norm(dim=-1), pre_slope.norm(dim=-1)
            share, top = top_energy_share(norm.square())
            ratio = torch.quantile(norm / pre_norm.clamp_min(1e-12), ratio_quantiles)
            rewrite = ((slope - pre_slope).square().sum() / pre_slope.square().sum().clamp_min(1e-30)).sqrt()
            old_mean, old_std = policy_moments(old_actor, obs)
            mean, std = actor(obs)
            move = torch.tanh(mean) - torch.tanh(old_mean)
            # Rows the trust region excludes from the objective at the end of the update.
            excluded = (gaussian_kl(old_mean, old_std, mean, std) >= args.kl_bound).float().mean()
            top_inverted = inverted[rows[top]].float().mean() if posture else torch.full_like(share, torch.nan)
            return torch.stack((share, norm.median(), norm.mean(), row_cosine(slope, pre_slope).median(),
                                ratio[0], ratio[1], ratio[2], rewrite, row_cosine(move, pre_slope).median(),
                                excluded, top_inverted))

        def gain_instrument(behaviour):
            """Realised and predicted gain of the previous update, on the rollout its result collected.

            Called after the insert, where previous_actor is the policy before that update, old_actor = actor
            the one after it (this rollout's behaviour policy), pre_critic the critic that update started from
            and target_critic = critic the one it ended with; none of them has been fit on these rows.
              realised   mean_j (A_j - mean A) w_j,  w_j = log pi_new(u_j|s_j) - log pi_old(u_j|s_j)
                         - KL(pi_new || pi_old)(s_j) (its mean over u: baseline error cannot leak), with u_j
                         the behaviour-time pre-tanh action (stored mean, std and noise), pi_old read at the
                         behaviour-time normalized observation, A_j = G_j - mean_k Q(s_j, a_k), a_k ~ pi_new,
                         and G the soft lambda = --diagnostic-lambda return (realised_hard: the entropy-free
                         one). realised_se is its lane standard error.
              predicted  mean_j [Q(s_j, a_j) - Q(s_j, tanh(mu_old + sigma_old eps_j))]: same rows, same noise.
              mean_part  the same quantities when only the mean moves (pi_old's mean under pi_new's std).
            Units: scaled return per step; (1 - gamma) (J_new - J_old) to first order.
            """
            obs, noise, actions = observations(rollout_rows), rollout_noise, rollout_actions
            mean, std = behaviour["mean"].flatten(0, 1), behaviour["std"].flatten(0, 1)
            old_mean, old_std = policy_moments(previous_actor, behaviour["norm_obs"].flatten(0, 1))
            log_ratio = centered_log_ratio(noise, mean, std, old_mean, old_std)
            samples = args.diagnostic_samples
            draws = torch.randn((samples,) + mean.shape, device=device, generator=instrument_generator)
            baseline = target_critic.value(obs.expand(samples, -1, -1), torch.tanh(mean + std * draws)).mean(0)
            terms = realised_gain_terms(diagnostic_targets - baseline, log_ratio)
            old_actions = torch.tanh(old_mean + old_std * noise)
            mean_only_actions = torch.tanh(old_mean + std * noise)
            pre_value = pre_critic.value(obs, actions)
            return torch.stack((
                terms.mean(), lane_standard_error(terms, N),
                realised_gain_terms(diagnostic_targets - baseline,
                                    centered_log_ratio(noise, mean, std, old_mean, std)).mean(),
                realised_gain_terms(hard_targets - baseline, log_ratio).mean(),
                (pre_value - pre_critic.value(obs, old_actions)).mean(),
                (target_critic.value(obs, actions) - target_critic.value(obs, old_actions)).mean(),
                (pre_value - pre_critic.value(obs, mean_only_actions)).mean()))

        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16)

        def encode_targets():
            """f'(s, a) for all rollout rows under the (just refreshed) target critic."""
            target_features.copy_(target_critic.features(observations(rollout_rows), rollout_actions))

        # Arms swap whole loss functions here; at default flags the two below are v60's own.
        if args.reward_latents:
            critic_loss = reward_latent_critic_loss
        if condition_slope is not None or actor_critic is not critic or args.kl_mode != "gate":
            actor_loss = arm_actor_loss
        if args.compile:
            if args.aux_every > 1:
                categorical_critic_loss = graph_compile(categorical_critic_loss)
            encode_targets = graph_compile(encode_targets)
            bootstrap_values = graph_compile(bootstrap_values)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        def uniform_rows():
            """Uniform fresh-rollout rows from the default CUDA generator (graph-safe)."""
            return torch.minimum((torch.rand(M, device=device) * B).long(), rollout_rows[-1])

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            with autocast():
                loss, metrics = critic_loss(uniform_rows())
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            critic_sums.add_(metrics)

        def actor_step():
            actor_optimizer.zero_grad(set_to_none=False)
            with autocast():
                loss, metrics = actor_loss(uniform_rows())
            # The critic is only differentiated through its action input.
            loss.backward(inputs=actor_parameters)
            grad_sums[0] += nn.utils.clip_grad_norm_(actor_network, args.max_grad_norm, foreach=True)
            actor_optimizer.step()
            actor_sums.add_(metrics)

        def categorical_critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            with autocast():
                loss, metrics = categorical_critic_loss(uniform_rows())
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            # Python-level only: under capture the recorded optimizer kernels simply omit these parameters.
            with withheld_gradients(aux_only_parameters):
                critic_optimizer.step()
            critic_sums.add_(metrics)

        def ema_tracked(step):
            """The critic step followed, in the same captured graph, by one step of the actor's weight EMA."""
            def tracked():
                step()
                with torch.no_grad():
                    torch._foreach_lerp_(ema_parameters, critic_parameters, ema_weight)
            return tracked

        # The full step first: it is the one that allocates every critic gradient during the capture's warmup.
        critic_steps = [critic_step] + ([categorical_critic_step] if args.aux_every > 1 else [])
        if ema_parameters:
            critic_steps = [ema_tracked(step) for step in critic_steps]

        update_steps = None  # captured lazily, once the first rollout populated the buffers
        # One Triton kernel; a cudagraph-tree wrapper would fault the manually captured update graphs.
        gae_fn = get_gae_fn(compiled=args.compile, mode="default", explicit_next_values=True)
        critic_per_actor = args.critic_updates // args.actor_updates

        host_actor = make_host_mirror(actor.net, args.num_envs)
        writer.add_text("host_actor", f"fused={host_actor.fused} fallback={host_actor.fallback_reason}")
        sampler = action_noise_rng(args.seed, args.rng_stream)
        sample_actions = HostSquashedGaussian(args.num_envs, action_dim)

        def act(observations):
            action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return action

        eval_actor = make_host_mirror(actor.net, 1) if args.eval_interval else None
        eval_env = gym.make(args.env_id) if args.eval_interval else None
        if eval_env is not None:
            resources.callback(eval_env.close)

        def evaluate(episode, replicate=0):
            """One mean-action episode, observations normalized with environment 0's statistics (shared: the
            pooled ones as they stand, which the next rollout would freeze). Its own env, host mirror and
            reset seed: no training stream is read (replicate 0 is v60's episode)."""
            eval_actor.refresh()
            if shared_norm is None:
                mean = obs_norm.means[0]
                inv_std = 1.0 / np.sqrt(obs_norm.variances[0] + obs_norm.epsilon)
            else:
                mean, inv_std = shared_norm.scale()
            raw, _ = eval_env.reset(seed=args.seed + 10_000 + episode + 1_000_000 * replicate)
            total, saturated, steps, done = 0.0, 0.0, 0, False
            while not done:
                if shared_norm is None:
                    normalized = np.clip((raw - mean) * inv_std, -args.obs_clip, args.obs_clip).astype(np.float32)
                else:
                    normalized = frozen_normalize(raw, mean, inv_std, args.obs_clip)
                action = np.tanh(np.split(eval_actor(normalized[None]), 2, axis=-1)[0][0])
                raw, reward, terminated, truncated, _ = eval_env.step(action)
                total += float(reward)
                saturated += float((np.abs(action) > 0.99).mean())
                steps += 1
                done = terminated or truncated
            return total, saturated / steps

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"raw_obs": obs_shape, "raw_next_obs": obs_shape, "actions": (action_dim,),
                                           "noise": (action_dim,), "mean": (action_dim,), "std": (action_dim,),
                                           "norm_obs": obs_shape})
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape, clip=args.obs_clip)
        # --obs-norm shared: obs_norm stays the per-lane accumulator (and normalizes the warmup); what the
        # recorded rollouts and the update see is its pooled statistic as of each rollout's start.
        shared_norm = SharedObsNorm(obs_norm) if args.obs_norm == "shared" else None
        step_offset = iteration_offset = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            actor.load_state_dict(checkpoint["actor_state"])
            critic.load_state_dict(checkpoint["critic"])
            target_critic.load_state_dict(checkpoint["critic"])
            if ema_parameters:
                # A v65 ema checkpoint carries its average; any other starts it at the checkpointed critic.
                actor_critic.load_state_dict(checkpoint.get("ema_critic", checkpoint["critic"]))
            old_actor.load_state_dict(checkpoint["old_actor"])
            previous_actor.load_state_dict(checkpoint["previous_actor"])
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            # In place: the native normalizer holds pointers to these arrays. Lane rows in either mode (a v60
            # checkpoint's own lanes, a shared one's pooled rows); --obs-norm shared pools them at each freeze.
            np.copyto(obs_norm.means, checkpoint["obs_means"])
            np.copyto(obs_norm.variances, checkpoint["obs_variances"])
            np.copyto(obs_norm.counts, checkpoint["obs_counts"])
            step_offset, iteration_offset = int(checkpoint["global_step"]), int(checkpoint["iteration"])
            host_actor.refresh()
            writer.add_text("fork", f"resume={args.resume} step={step_offset} iteration={iteration_offset} "
                                    f"fork_ent_target_mult={args.fork_ent_target_mult} "
                                    f"fork_iterations={args.fork_iterations}")

        def normalizer_state():
            """Saved (means, variances, counts), one row per lane: the lanes' own, or under --obs-norm shared
            the pooled moments in every row (what a lane-indexed reader should normalize with)."""
            if shared_norm is None:
                return obs_norm.means.copy(), obs_norm.variances.copy(), obs_norm.counts.copy()
            return shared_norm.lane_state()

        def save_checkpoint(path, global_step, iteration):
            """Full training state, plus the snapshot keys scripts/landscape.py and fork_eval.py read."""
            obs_means, obs_variances, obs_counts = normalizer_state()
            torch.save({"actor_state": actor.state_dict(), "critic": critic.state_dict(),
                        "old_actor": old_actor.state_dict(), "previous_actor": previous_actor.state_dict(),
                        "actor_optimizer": actor_optimizer.state_dict(),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "actor_net": actor.net.state_dict(), "obs_means": obs_means,
                        "obs_variances": obs_variances, "obs_counts": obs_counts,
                        "obs_epsilon": obs_norm.epsilon, "global_step": global_step, "iteration": iteration,
                        "temperature": actor.temperature().tolist(), "target_entropy": float(target_entropy),
                        "args": vars(args),
                        **({"ema_critic": actor_critic.state_dict()} if ema_parameters else {})}, path)

        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations).reshape((args.num_envs,) + space.shape)

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
            # Warmup hands back only normalized observations; they were normalized with the current
            # statistics, so inverting them recovers the raw ones (exact unless clipped).
            raw_current = (next_obs_np * np.sqrt(obs_norm.variances + obs_norm.epsilon)
                           + obs_norm.means).astype(np.float32)
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            raw_current = np.array(raw_obs, dtype=np.float32)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        raw_next = np.empty_like(raw_current)
        global_step += step_offset
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step
        pending_snapshots = [step for step in args.snapshot_at if step > global_step]
        pending_checkpoints = [step for step in args.checkpoint_at if step > global_step]
        last_iteration = iteration_offset + args.num_iterations

        for iteration in range(iteration_offset + 1, last_iteration + 1):
            fork_index = iteration - iteration_offset
            multiplier = args.ent_target_mult
            if args.fork_ent_target_mult >= 0 and fork_index <= args.fork_iterations:
                multiplier = args.fork_ent_target_mult
            target_entropy.fill_(-multiplier * action_dim)
            host_actor.refresh()
            if shared_norm is not None:
                # The statistic of this whole iteration: everything counted up to here (the warmup and every
                # earlier rollout). The pending observation was normalized with the previous one.
                shared_norm.freeze()
                next_obs_np = shared_norm.normalize(raw_current)
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(action.reshape((args.num_envs,) + space.shape))
                with timer.span("normalize_transfer", use_cuda=False):
                    next_obs_np, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    if shared_norm is not None:
                        # The call above only counted the step (final observations before resets, as in v60).
                        next_obs_np = shared_norm.normalize(raw_obs)
                    np.copyto(raw_next, raw_obs)
                    for index in np.flatnonzero(np.logical_or(terms, truncs)):
                        raw_next[index] = infos["final_observation"][index]
                    transfer.push(step, np.asarray(raw_reward) * args.reward_scale, terms, truncs,
                                  raw_obs=raw_current, raw_next_obs=raw_next, actions=action,
                                  noise=sample_actions.noise, mean=sample_actions.mean,
                                  std=sample_actions.std, norm_obs=obs_step)
                    np.copyto(raw_current, raw_obs)
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

            with timer.span("insert"), torch.no_grad():
                batch = transfer.upload()
                rollout_obs.copy_(batch.fields["raw_obs"].flatten(0, 1))
                rollout_next_obs.copy_(batch.fields["raw_next_obs"].flatten(0, 1))
                rollout_actions.copy_(batch.fields["actions"].flatten(0, 1))
                rollout_noise.copy_(batch.fields["noise"].flatten(0, 1))
                rollout_rewards.copy_(batch.rewards.flatten())
                rollout_continuing.copy_(1.0 - batch.terminations.flatten())
                rollout_truncated.copy_(batch.truncations.flatten())
                if shared_norm is None:
                    obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                    obs_inv_std.copy_(torch.from_numpy(
                        (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                else:
                    # Every lane's row is the statistic the rollout was collected under (still frozen).
                    obs_mean.copy_(torch.from_numpy(shared_norm.mean))
                    obs_inv_std.copy_(torch.from_numpy(shared_norm.inv_std))
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(previous_actor.parameters()), list(old_actor.parameters()))
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

            if args.actor_critic == "frozen":
                with torch.no_grad():
                    torch._foreach_copy_(list(actor_critic.parameters()), critic_parameters)

            cuts = torch.maximum(rollout_truncated.view(time_shape), rollout_ends)
            with torch.no_grad():
                # The episode survives step t -> t + 1 inside the rollout unless it terminated, truncated, or
                # the rollout ended at t.
                survives = (rollout_continuing.view(time_shape) * (1.0 - cuts))
                chain = torch.ones_like(survives)
                for k in range(H):
                    chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
                    rollout_chain[k].copy_(chain.flatten())

            def refresh_targets(lam, out, soft=True):
                """Soft lambda-returns G_t = r_t + gamma [(1 - lam) V'_{t+1} + lam (G_{t+1} + T H^_{t+1})].

                The lambda-branch's entropy bonus at s_{t+1} is the K-sample expected one (unbiased for the
                taken action's -T log pi, lower variance). It is folded into the reward where the chain
                continues; cut chains (truncation, rollout end) bootstrap from V' alone; terminations
                bootstrap nothing.
                """
                next_values, bonus = bootstrap_values(rollout_rows)
                continuing = rollout_continuing.view(time_shape)
                rewards = rollout_rewards.view(time_shape)
                if soft:
                    rewards = rewards + args.gamma * lam * continuing * (1.0 - cuts) * bonus.view(time_shape)
                else:  # the benchmark's return: no entropy anywhere, bootstrapping from E_pi Q alone
                    next_values = next_values - bonus
                out.copy_(lambda_returns(gae_fn, rewards, next_values.view(time_shape), 1.0 - continuing, cuts,
                                         args.gamma, lam).flatten())

            with timer.span("targets"), torch.no_grad():
                refresh_targets(args.diagnostic_lambda, diagnostic_targets)
                refresh_targets(args.diagnostic_lambda, hard_targets, soft=False)

            with timer.span("diagnostics"), torch.no_grad():
                # The actor still equals the rollout's behavior policy (no update yet this iteration).
                diagnostic_rows = torch.randperm(B, device=device, generator=diagnostic_generator)[: args.diagnostic_rows]
                diagnostics = critic_diagnostics(diagnostic_rows)
                target_temperature, ids_metrics = information_price(diagnostic_rows)
                probe_metrics = regret_probe()
                if iteration > 1 and not actor.dual:
                    # Iteration 1's critic has never been fit, so its slope carries no information yet.
                    budget = args.dual_step_per_iteration
                    actor.log_temperature.add_(
                        (target_temperature.log() - actor.log_temperature).clamp(-budget, budget))

            instrument_due = iteration % args.instrument_interval == 0
            instrument_metrics, inverted = {}, None
            with timer.span("instrument"), torch.no_grad():
                if posture:
                    inverted, posture_metrics = posture_instrument()
                    instrument_metrics.update(
                        {f"posture/{name}": posture_metrics[i] for i, name in enumerate(POSTURE_METRICS)})
                if pre_critic_iteration == iteration - 1:
                    # pre_critic still holds the critic the previous update started from.
                    gain_metrics = gain_instrument(batch.fields)
                    instrument_metrics.update(
                        {f"gain/{name}": gain_metrics[i] for i, name in enumerate(GAIN_METRICS)})
                if instrument_due:
                    torch._foreach_copy_(list(pre_critic.parameters()), critic_parameters)
                    pre_critic_iteration = iteration

            profiler = None
            if iteration == args.profile_iteration:
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                profiler.__enter__()
            with timer.span("update"):
                if update_steps is None:
                    if args.compile:
                        graphs = capture_steps([*critic_steps, actor_step],
                                               optimizers=[critic_optimizer, actor_optimizer],
                                               state_tensors=[critic_sums, actor_sums, grad_sums,
                                                              *ema_parameters])
                        update_steps = [graph.replay for graph in graphs]
                    else:
                        update_steps = [*critic_steps, actor_step]
                # With --aux-every 1 both critic names are the one (v60) critic step.
                critic_update, categorical_update, actor_update = (
                    update_steps[0], update_steps[-2], update_steps[-1])
                critic_sums.zero_()
                actor_sums.zero_()
                grad_sums.zero_()
                for update in range(1, args.critic_updates + 1):
                    if (update - 1) % args.target_refresh == 0:
                        with torch.no_grad():
                            if update > 1:
                                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                            refresh_targets(args.gve_lambda, rollout_targets)
                            encode_targets()
                    (critic_update if (update - 1) % args.aux_every == 0 else categorical_update)()
                    if update % critic_per_actor == 0:
                        actor_update()

            if profiler is not None:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                with open(f"runs/{run_name}/update_profile.txt", "w") as handle:
                    handle.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40))
            with timer.span("instrument"), torch.no_grad():
                if instrument_due:
                    instrument_rows = torch.randperm(B, device=device,
                                                     generator=instrument_generator)[: args.instrument_rows]
                    conditioning = conditioning_instrument(instrument_rows, inverted)
                    instrument_metrics.update(
                        {f"cond/{name}": conditioning[i] for i, name in enumerate(CONDITIONING_METRICS)})
                    if posture:
                        split = split_instrument(inverted, batch.fields["std"].flatten(0, 1))
                        instrument_metrics.update(
                            {f"split/{name}": split[i] for i, name in enumerate(SPLIT_METRICS)})
            logged = gather_metrics({
                **{f"losses/{name}": critic_sums[i] / critic_metric_steps[i] for i, name in enumerate(CRITIC_METRICS)},
                **{f"policy/{name}": actor_sums[i] / args.actor_updates for i, name in enumerate(ACTOR_METRICS)},
                **{f"diagnostics/{name}": diagnostics[i] for i, name in enumerate(DIAGNOSTIC_METRICS)},
                **{f"ids/{name}": ids_metrics[i] for i, name in enumerate(IDS_METRICS)},
                **{f"probe/{name}": probe_metrics[i] for i, name in enumerate(PROBE_METRICS)},
                **{f"ids/temperature_{i}": value for i, value in enumerate(actor.temperature())},
                "grad/actor_preclip_norm": grad_sums[0] / args.actor_updates,
                "grad/critic_preclip_norm": grad_sums[1] / args.critic_updates,
            })
            if any(not np.isfinite(value) for value in logged.values()):
                raise FloatingPointError("nonfinite REPPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            if instrument_metrics:
                for name, value in gather_metrics(instrument_metrics).items():
                    if np.isfinite(value):  # an empty posture group has no mean; instruments never stop a run
                        writer.add_scalar(name, value, global_step)
            if args.eval_interval and (iteration % args.eval_interval == 0 or iteration == last_iteration):
                with timer.span("eval", use_cuda=False):
                    eval_return, eval_saturation = evaluate(iteration)
                    eval_returns = [eval_return] + [evaluate(iteration, replicate)[0]
                                                    for replicate in range(1, args.eval_episodes)]
                writer.add_scalar("eval/deterministic_return", eval_return, global_step)
                writer.add_scalar("eval/det_return_mean", float(np.mean(eval_returns)), global_step)
                writer.add_scalar("eval/det_return_min", min(eval_returns), global_step)
                writer.add_scalar("eval/deterministic_saturation", eval_saturation, global_step)
            while pending_snapshots and global_step >= pending_snapshots[0]:
                pending_snapshots.pop(0)
                # Everything the oracle needs to rebuild the acting policy and the critic that judges it.
                obs_means, obs_variances, _ = normalizer_state()
                torch.save({"actor_net": actor.net.state_dict(), "critic": critic.state_dict(),
                            "obs_means": obs_means, "obs_variances": obs_variances,
                            "obs_epsilon": obs_norm.epsilon, "global_step": global_step, "iteration": iteration,
                            "temperature": actor.temperature().tolist(), "args": vars(args)},
                           f"runs/{run_name}/snapshot_{global_step}.pt")
            while pending_checkpoints and global_step >= pending_checkpoints[0]:
                pending_checkpoints.pop(0)
                save_checkpoint(f"runs/{run_name}/checkpoint_{global_step}.pt", global_step, iteration)
            if args.resume and fork_index in (args.fork_iterations, args.num_iterations):
                save_checkpoint(f"runs/{run_name}/fork_{fork_index}.pt", global_step, iteration)
            writer.add_scalar("fork/target_entropy", -multiplier * action_dim, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step
    finally:
        resources.close()


if __name__ == "__main__":
    main()
