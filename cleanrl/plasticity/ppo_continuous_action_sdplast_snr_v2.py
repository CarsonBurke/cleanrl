# SDPLAST-SNR v2 -- state-dependent plasticity from gradient SIGNAL-TO-NOISE.
# =====================================================================================
# WHAT v1 SETTLED (run HalfCheetah-v4__sdplast_v1__1, 8M, seed 1)
#   v1 gated each perceptron's per-sample weight gradient by an exact per-(sample,
#   unit) IDBD/hypergradient signal: raise plasticity where this sample's proposed
#   edit agrees with the direction the unit has been travelling (Adam's m/sqrt(v)).
#   Result: 7455.0 +-292 final-20 against the baseline's 7468.1 +-123. Matched
#   steps +82 @4M, +211 @6M, +197 @7M -- all inside the 423-point one-seed noise
#   floor. But the gates were fully engaged: data_std 0.399, rate_std 0.198,
#   logit_abs 0.86, versus statedynlr_v8's realized rate_std of 0.013. So the
#   mechanism ran 15-30x harder than any predecessor and bought nothing.
#
#   Two diagnoses, both actionable:
#   (a) AGREEMENT IS SELF-REFERENTIAL. Adam's m has a ~7-minibatch half-life and
#       a rollout is 320 minibatches, so H is essentially the CURRENT rollout's
#       mean gradient. "Upweight samples agreeing with H" is therefore shrinkage
#       of a batch toward its own mean: it lowers update variance while adding
#       zero information, and it is BIASED for the mean it is estimating. Adam
#       already normalizes per-coordinate scale, so such a reweighting is both
#       nearly free and nearly worthless.
#   (b) I CONSERVED A BUDGET THAT WAS NOT SCARCE. v1 subtracted the layer mean
#       from the gate logit, making the mechanism learning-rate-neutral by
#       construction, so it could only REALLOCATE plasticity. But at 7M the
#       baseline sits at clipfrac 0.019 and approx_kl 0.0032 against
#       clip_coef 0.2 -- roughly 10x trust-region headroom, critic EV 0.92. The
#       actor is under-driven, not conflict-limited. Nothing was fighting, so
#       there was nothing worth reallocating.
#
# THE CRITERION (this is the whole change)
#   A step size should not be set by "am I consistent with my own past" but by
#   "how reliable is my gradient estimate right now". For unit i, its per-sample
#   contribution is the augmented row r_t = delta_{t,i} * [x_t, 1]. Then
#       rbar_i  = mean_t r_t                (the signal)
#       ell_t   = |r_t - rbar_i|^2          (this sample's noise energy)
#       SNR_i   = |rbar_i|^2 / mean_t|r_t|^2   in [0, 1]
#   and the two factors follow:
#     1. ALLOCATION ACROSS STATES = inverse conditional variance. Weight sample t
#        by w_{t,i} ~ 1 / E[ell | s_t]. This is textbook GLS: under
#        heteroscedastic noise it is unbiased and has strictly lower variance
#        than uniform weighting (Gauss-Markov). Advantage noise in MuJoCo IS
#        strongly state-dependent, so there is real heteroscedasticity to exploit.
#     2. LEVEL PER UNIT = Wiener shrinkage on the measured SNR. A unit whose
#        gradient is mostly signal earns a larger step; one whose gradient is
#        mostly noise earns a smaller one. NOT normalized to mean one -- the
#        whole layer's plasticity is free to rise as its gradients become
#        coherent, which is the channel v1's conservation forbade.
#
# WHY NO HELD-OUT SPLIT IS NEEDED (and why v1 would have needed one)
#   Estimator variance is a property of the estimator: raising a unit's gate does
#   not change its measured noise, so there is no feedback loop to break. And
#   crucially, GLS is unbiased exactly when the weight is independent of the
#   sample's own noise realization. A weight predicted from STATE satisfies that;
#   a weight read off the realized residual does not. So the premise -- each
#   perceptron decides from its own state -- is precisely the condition that
#   makes this legal. That is why the head predicts log-noise from state and is
#   trained on the residual, rather than using the residual directly.
#
# HOW THE PERCEPTRON DECIDES
#   Each unit owns an affine readout over its own bounded, scale-free state (its
#   normalized pre-activation, its magnitude, and a rank-ctx_dim view of the
#   layer input): p_{t,i} = log-noise prediction, bounded by
#   p_cap = log(weight_max) so w = exp(-p) lands in [1/weight_max, weight_max]
#   automatically. It is trained by the heteroscedastic Gaussian NLL
#       L_gate = mean( p + ell_hat * exp(-p) ),   ell_hat = ell / mean_t(ell_i)
#   whose unique minimizer is exp(p) = E[ell_hat | s]. A well-posed regression
#   with a state-measurable target -- no injected surrogate gradient, no
#   z-scoring, no constructed mean-one target, no held-out data.
#
# GETTING delta WITHOUT AN EXTRA BACKWARD
#   ell needs delta = dL/dz, which exists only in the backward pass. Each site
#   adds an exact-zero (minibatch, width) `probe` parameter to its output, so
#   `probe.grad` IS delta after the ordinary loss.backward(): no hooks, no
#   retain_graph, no second network backward, one elementwise add. The gate heads
#   are then trained by a separate tiny backward over the heads alone.
#
# ADAM-SURVIVING APPLICATION (kept from v1, verified)
#   A per-row gradient rescale cancels inside Adam (statedynlr_v7: 1869 @1.58M),
#   so w is applied per-sample PRE-Adam (it changes the gradient direction, which
#   Adam cannot normalize away) and the level lam_i is applied to the realized
#   step POST-Adam, as a correction so a neutral unit steps bit-exactly.
#   Forward output stays bit-identical to nn.Linear, so the host actor mirror and
#   the rollout path are untouched.
#
# CALIBRATION, NOT A MAGIC SCALE
#   SNR has no natural absolute scale, so `snr_warmup` updates run with
#   lam == 1 exactly while a global mean SNR reference is measured, then frozen.
#   Afterwards lam_i = exp(L * tanh(log(sqrt(SNR_i)/snr_ref) / L)) with
#   L = log(lam_span): symmetric and smoothly bounded in LOG space, exactly 1 at
#   the reference, with no probability mass piling up at a rail. A hard clamp
#   instead lifted mean(lam) to 1.18 -- an 18% learning-rate increase smuggled
#   into the mechanism, which the LR controls would then explain for free. The
#   reference is frozen on the mean of sqrt(SNR), so the level is free to drift
#   up or down -- absolute, not conserved -- while still starting at 1.
#
# HYPOTHESIS
#   The binding constraint at 8M is estimator quality, not credit conflict.
#   Inverse-noise allocation is the unbiased variance reduction that agreement
#   weighting only pretended to be, and an SNR-driven level spends the 10x
#   trust-region headroom exactly where the gradient can be trusted -- which is
#   strictly better than the uniform LR increase that is this lineage's real
#   (and previously unacknowledged) source of gains.
#
# PASS: beats BOTH the baseline (7468 final-20) AND the uniform-LR controls
#   ppo_lrctl_5.1e-4 / ppo_lrctl_8.1e-4 at matched steps, with sdp/lam_std > 0.05
#   and the gate NLL falling.
# FAIL: >5% below baseline at two consecutive checkpoints, lam pinned at a rail,
#   NLL flat (no predictable heteroscedasticity => the premise is empty here).
# WATCH: sdp/<site>_w_std, sdp/<site>_lam_mean, sdp/<site>_lam_std,
#   sdp/<site>_snr, losses/gate_nll, losses/clipfrac, timing/update_s.
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
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import HostMLP
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


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
    """whether to capture videos of the agent performances"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # State-dependent plasticity from gradient SNR
    plastic_actor: bool = True
    """gate the plasticity of the actor's hidden units"""
    plastic_critic: bool = True
    """gate the plasticity of the critic's hidden units"""
    ctx_dim: int = 8
    """rank of the shared layer-context each unit reads alongside its own state"""
    weight_max: float = 2.0
    """log-noise cap; the normalized GLS weight lies in [1/weight_max^2, weight_max^2]"""
    gate_lr: float = 1e-3
    """learning rate of the per-unit log-noise predictors"""
    gate_every: int = 4
    """supervise the predictors every k minibatches; the noise structure is slow
    and the measurement costs ~2.3x an unsupervised update, so amortizing keeps
    the update-phase overhead at ~1.3x (+1.5% wall clock) for ~19.5k predictor
    updates over an 8M run"""
    gate_clip: float = 1.0
    """gradient-norm clip for plasticity parameters, kept off the network's clip"""
    lam_span: float = 1.5
    """post-Adam step level lies in (1/lam_span, lam_span), smoothly bounded.
    Deliberately narrow: because the reference is frozen, a systematic rise in
    SNR can drift the whole layer's level toward the ceiling, which is a UNIFORM
    learning-rate change and therefore indistinguishable from the ppo_lrctl
    arms. 1.5 keeps that worst case strictly inside their 1.7x-2.7x bracket, so
    a win cannot be an unlabelled LR tune. Watch sdp/*_lam_mean against
    sdp/*_lam_std: mean drift is an LR effect, dispersion is the mechanism."""
    lam_gain: float = 1.0
    """sensitivity of the level to log relative Wiener amplitude"""
    snr_beta: float = 0.99
    """EMA decay for each unit's measured gradient SNR"""
    snr_warmup: int = 100
    """updates run at lam == 1 while the frozen SNR reference is calibrated"""
    gls_weights: bool = True
    """apply the per-sample inverse-noise weight (factor 1)"""
    snr_level: bool = True
    """apply the per-unit SNR step level (factor 3)"""

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 4
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlasticLinear(nn.Linear):
    """Linear layer whose per-sample weight gradient is inverse-noise weighted by
    a per-unit predictor reading that unit's own state.

    Subclasses ``nn.Linear`` on purpose: the forward value is bit-identical, so
    ``HostMLP`` mirrors it and the rollout is unaffected. Under ``no_grad`` the
    gate machinery is skipped entirely, which is every inference path.

    Gradient contract, with delta = dL/dz the incoming pre-activation gradient:
        dL/dW_i = sum_t w_{t,i} * delta_{t,i} * x_t     GLS-weighted
        dL/db_i = sum_t w_{t,i} * delta_{t,i}           GLS-weighted
        dL/dx_t = W^T delta_t                          UNGATED
        probe.grad = delta                             exported for supervision
    ``lam`` (the per-unit level) is exported through a buffer because Adam
    cancels a per-row gradient rescale; the trainer applies it to the realized
    step instead. The predictor receives no gradient from the PPO loss at all --
    it is trained by its own NLL against the measured residual.
    """

    def __init__(self, in_features, out_features, args, minibatch_size):
        super().__init__(in_features, out_features)
        self.p_cap = float(np.log(args.weight_max))
        self.w_max = float(args.weight_max) ** 2
        self.w_min = 1.0 / self.w_max
        self.use_weights = bool(args.gls_weights)
        # Each unit's log-noise predictor: an affine readout over its own
        # bounded state features plus a rank-ctx_dim view of the layer input.
        # All zero at init => p == 0 => w == 1 => exactly the baseline.
        self.ctx_proj = nn.Parameter(torch.empty(int(args.ctx_dim), in_features))
        nn.init.orthogonal_(self.ctx_proj, 1.0)
        self.ctx_read = nn.Parameter(torch.zeros(out_features, int(args.ctx_dim)))
        self.w_state = nn.Parameter(torch.zeros(out_features))
        self.w_mag = nn.Parameter(torch.zeros(out_features))
        self.p_bias = nn.Parameter(torch.zeros(out_features))
        # Exact-zero addend whose gradient is delta. Not an optimizer parameter.
        self.probe = nn.Parameter(torch.zeros(minibatch_size, out_features))
        self.register_buffer("lam", torch.ones(out_features))
        self.register_buffer("snr", torch.zeros(out_features))
        self.register_buffer("stats", torch.zeros(4))

    def gate_parameters(self):
        return [self.ctx_proj, self.ctx_read, self.w_state, self.w_mag, self.p_bias]

    def state_features(self, x, z):
        """The perceptron's own state: bounded and scale-free."""
        zn = z * torch.rsqrt(z.square().mean(0, keepdim=True) + 1e-8)
        zs = zn.square()
        return zn / (1.0 + zn.abs()), zs / (1.0 + zs) - 0.5

    def predict_log_noise(self, x, z):
        """p_{t,i}: unit i's predicted log noise energy in state t.

        Centered per unit inside the graph: the predictor learns only the STATE
        dependence of the noise, never its level (that is lam's job, and lam is
        the factor that survives Adam). Bounded by p_cap = log(weight_max), so
        exp(-p) lands in [1/weight_max, weight_max] before renormalization and
        in [1/weight_max^2, weight_max^2] after it.
        """
        f_state, f_mag = self.state_features(x, z)
        ctx = torch.tanh(F.linear(x, self.ctx_proj))
        raw = (self.p_bias + self.w_state * f_state + self.w_mag * f_mag
               + F.linear(ctx, self.ctx_read))
        raw = raw - raw.mean(0, keepdim=True)
        return self.p_cap * torch.tanh(raw / self.p_cap)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return F.linear(x, self.weight, self.bias)
        xd = x.detach()
        wd, bd = self.weight.detach(), self.bias.detach()
        z = F.linear(xd, wd, bd)

        if self.use_weights:
            with torch.no_grad():
                p = self.predict_log_noise(xd, z)
                # GLS weight, renormalized to mean one per unit so that the
                # LEVEL is owned solely by lam (which survives Adam), then
                # clamped to the documented envelope.
                w = torch.exp(-p)
                w = (w / w.mean(0, keepdim=True)).clamp(self.w_min, self.w_max)
        else:
            w = torch.ones_like(z)
        self.stats[0] = w.std()

        # Value is exactly F.linear(x, W, b): the second term is w * 0 and the
        # probe is exactly zero, so the forward is bit-identical to nn.Linear.
        y = F.linear(x, wd, bd)
        y = y + w * (F.linear(xd, self.weight, self.bias) - z)
        return y + self.probe


class Agent(nn.Module):
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

        def hidden(in_features, out_features, plastic):
            if not plastic:
                return nn.Linear(in_features, out_features)
            return PlasticLinear(in_features, out_features, args, args.minibatch_size)

        self.critic = nn.Sequential(
            layer_init(hidden(observation_dim, 64, args.plastic_critic)), nn.Tanh(),
            layer_init(hidden(64, 64, args.plastic_critic)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(hidden(observation_dim, 64, args.plastic_actor)), nn.Tanh(),
            layer_init(hidden(64, 64, args.plastic_actor)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        self.plastic_sites = [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, PlasticLinear)
        ]

    def gate_parameters(self):
        return [p for _, module in self.plastic_sites for p in module.gate_parameters()]

    def probe_parameters(self):
        return [module.probe for _, module in self.plastic_sites]

    def network_parameters(self):
        excluded = {id(p) for p in self.gate_parameters()}
        excluded.update(id(p) for p in self.probe_parameters())
        return [p for p in self.parameters() if id(p) not in excluded]

    @torch.no_grad()
    def site_activations(self, observations):
        """(input, pre-activation) at each plastic site, on the no_grad fast path.

        Order matches ``plastic_sites``: the critic trunk then the actor trunk.
        """
        pairs = []
        for trunk in (self.critic, self.actor):
            h = observations
            for module in trunk:
                if isinstance(module, PlasticLinear):
                    pairs.append((h, F.linear(h, module.weight, module.bias)))
                h = module(h)
        return pairs

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training retains native samples."""
        alpha, beta, value = self.get_policy_and_value(x)
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
        return action, logprob, entropy, value


def gate_supervision(agent, observations, args):
    """Measure each perceptron's gradient noise, then regress its predictor on it.

    Called after ``loss.backward()``, so ``probe.grad`` holds delta = dL/dz. For
    unit i the augmented per-sample contribution is r_t = delta_{t,i} * [x_t, 1]:
        rbar  = mean_t r_t                          signal
        ell_t = |r_t - rbar|^2                      noise energy of sample t
        SNR   = |rbar|^2 / mean_t |r_t|^2   in [0, 1]
    ``ell`` is divided by its per-unit mean, so the predictor learns only the
    STATE dependence of the noise (the per-unit level is lam's job) and the
    heteroscedastic NLL ``p + ell_hat * exp(-p)`` is minimized at
    ``exp(p) = E[ell_hat | s]``.
    """
    activations = agent.site_activations(observations)
    total = observations.new_zeros(())
    for (name, layer), (x, z) in zip(agent.plastic_sites, activations):
        delta = layer.probe.grad
        if delta is None:
            continue
        with torch.no_grad():
            rows = delta.shape[0]
            energy_scale = x.square().sum(1, keepdim=True) + 1.0     # |[x, 1]|^2
            squared = delta.square()
            row_energy = squared * energy_scale                      # |r_t|^2
            weight_mean = (delta.transpose(0, 1) @ x) / rows          # rbar weight part
            bias_mean = delta.mean(0)                                 # rbar bias part
            cross = F.linear(x, weight_mean, bias_mean)               # <[x,1], rbar>
            signal = weight_mean.square().sum(1) + bias_mean.square()  # |rbar|^2
            ell = (row_energy - 2.0 * delta * cross + signal).clamp_min(0.0)
            ell_hat = ell / (ell.mean(0, keepdim=True) + 1e-12)
            snr = signal / (row_energy.mean(0) + 1e-12)
            layer.snr.mul_(args.snr_beta).add_(snr, alpha=1.0 - args.snr_beta)
            layer.stats[3] = snr.mean()
        prediction = layer.predict_log_noise(x, z)
        total = total + (prediction + ell_hat * torch.exp(-prediction)).mean()
    return total


class PlasticityStepper:
    """The three update-path duties the compiled PPO loss cannot own.

    1. ``levels``: convert each unit's EMA SNR into its step level. SNR has no
       absolute scale, so the first ``snr_warmup`` updates run at lam == 1 while
       a global reference is measured, then frozen. Afterwards
       ``lam_i = exp(L * tanh(log(sqrt(SNR_i)/snr_ref) / L))``, L = log(span):
       exactly 1 at the reference, bounded in log space so nothing rails, and
       NOT renormalized to mean one -- v1 proved that conserving the budget
       makes the mechanism inert on a task with 10x trust-region headroom.
    2. ``apply_levels``: Adam is invariant to a per-row gradient rescale, so lam
       is applied to the REALIZED step as the correction
       ``w_i += (lam_i - 1) * (w_i^after - w_i^before)``, which leaves Adam's
       moments untouched and is bit-exact for a neutral unit.
       ``statedynlr_v7`` proved the pre-Adam alternative collapses (1869 @1.58M).
    3. ``clear_probes``: the delta exporters must start every backward at zero.
    """

    def __init__(self, sites, args):
        self.layers = [module for _, module in sites]
        self.args = args
        self.enabled = bool(args.snr_level) and bool(self.layers)
        self.params = [p for layer in self.layers for p in (layer.weight, layer.bias)]
        self.snapshots = [torch.empty_like(p) for p in self.params]
        self.probes = [layer.probe for layer in self.layers]
        self.updates = 0
        self.reference = None

    def clear_probes(self):
        for probe in self.probes:
            probe.grad = None

    @torch.no_grad()
    def levels(self):
        """Publish lam from the measured SNR; calibrate the reference first.

        The Wiener amplitude is sqrt(SNR), so the reference is the mean of
        sqrt(SNR) across every gated unit, NOT sqrt of the mean SNR. Calibrating
        on the mean SNR would leave mean(lam) = E[sqrt(x)]/sqrt(E[x]) < 1 by
        Jensen -- a systematic learning-rate CUT, exactly backwards on a task
        with 10x trust-region headroom. This way mean(lam) is 1 at calibration
        and free to drift afterwards.
        """
        self.updates += 1
        if not self.enabled:
            return
        if self.updates <= self.args.snr_warmup:
            if self.updates == self.args.snr_warmup:
                pooled = torch.cat([layer.snr.clamp_min(0.0).sqrt() for layer in self.layers])
                self.reference = pooled.mean().clamp_min(1e-12).clone()
            return
        span = float(np.log(self.args.lam_span))
        for layer in self.layers:
            amplitude = layer.snr.clamp_min(1e-30).sqrt() / self.reference
            layer.lam.copy_(torch.exp(span * torch.tanh(
                self.args.lam_gain * amplitude.log() / span)))
            layer.stats[1] = layer.lam.mean()
            layer.stats[2] = layer.lam.std()

    def stash(self):
        if self.enabled:
            torch._foreach_copy_(self.snapshots, self.params)

    @torch.no_grad()
    def apply_levels(self):
        if not self.enabled:
            return
        for index, layer in enumerate(self.layers):
            offset = layer.lam - 1.0
            for param, snapshot, gain in (
                (layer.weight, self.snapshots[2 * index], offset.unsqueeze(1)),
                (layer.bias, self.snapshots[2 * index + 1], offset),
            ):
                # snapshot: w_before -> -(step) -> -(lam-1)*step, then subtract.
                snapshot.sub_(param).mul_(gain)
                param.sub_(snapshot)


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
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
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.batch_size % args.minibatch_size:
        raise ValueError("the delta probes require every minibatch to have the same size")
    if args.norm_adv and args.minibatch_size < 2:
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if args.ctx_dim <= 0 or args.weight_max <= 1.0:
        raise ValueError("ctx_dim must be positive and weight_max must exceed one")
    if args.lam_span <= 1.0 or args.lam_gain <= 0.0:
        raise ValueError("lam_span must exceed one and lam_gain must be positive")
    if not (0.0 < args.snr_beta < 1.0) or args.snr_warmup < 1:
        raise ValueError("snr_beta must lie in (0, 1) and snr_warmup must be positive")
    if args.gate_every < 1:
        raise ValueError("gate_every must be positive")
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
                      matmul_precision="highest", allow_tf32=False)
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
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        writer.add_text("plasticity", "per-unit log-noise predictor -> GLS per-sample weight (pre-Adam) "
                                      "x SNR step level (post-Adam); NLL on the measured residual")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        gate_params = agent.gate_parameters()
        net_params = agent.network_parameters()
        optimizer = optim.Adam(
            [{"params": net_params, "lr": args.learning_rate},
             {"params": gate_params, "lr": args.gate_lr}],
            lr=args.learning_rate, eps=1e-5, fused=True,
        )
        plasticity = PlasticityStepper(agent.plastic_sites, args)
        print(f"plastic sites: {[name for name, _ in agent.plastic_sites]} | "
              f"gate params: {sum(p.numel() for p in gate_params)}")
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        def gate_model(observations):
            return gate_supervision(agent, observations, args)

        gate_scale = float(args.gate_every)
        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            gate_model = torch.compile(gate_model, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        host_actor = HostMLP(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)

        def act(observations):
            native, action = sample_beta_actions_host(host_actor(observations), action_low, action_high, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * (args.batch_size // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        gate_losses = torch.empty(max_updates, device=device)
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
                fraction = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
                optimizer.param_groups[1]["lr"] = fraction * args.gate_lr
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
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        observations = b_obs[indices]
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            observations, b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        supervise = updates % args.gate_every == 0
                        optimizer.zero_grad(set_to_none=True)
                        # Unconditional: probe.grad lives in the cudagraph pool,
                        # so leaving it set makes the next backward accumulate
                        # into a tensor a later replay has already overwritten.
                        plasticity.clear_probes()
                        loss.backward()
                        if supervise:
                            # probe.grad now holds delta; measure the noise and
                            # train each perceptron's predictor on it. Scaled by
                            # gate_every so the predictor's effective learning
                            # rate is independent of the supervision cadence.
                            gate_loss = gate_model(observations) * gate_scale
                            gate_loss.backward()
                            gate_losses[updates].copy_(gate_loss.detach())
                        else:
                            gate_losses[updates].copy_(gate_losses[max(updates - 1, 0)])
                        nn.utils.clip_grad_norm_(net_params, args.max_grad_norm)
                        if gate_params:
                            nn.utils.clip_grad_norm_(gate_params, args.gate_clip)
                        plasticity.levels()
                        plasticity.stash()
                        optimizer.step()
                        plasticity.apply_levels()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = {
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "losses/gate_nll": gate_losses[:updates].mean() / gate_scale,
            }
            for name, module in agent.plastic_sites:
                site = name.replace(".", "_")
                logged[f"sdp/{site}_w_std"] = module.stats[0]
                logged[f"sdp/{site}_lam_mean"] = module.stats[1]
                logged[f"sdp/{site}_lam_std"] = module.stats[2]
                logged[f"sdp/{site}_snr"] = module.stats[3]
            logged = gather_metrics(logged)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
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
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
