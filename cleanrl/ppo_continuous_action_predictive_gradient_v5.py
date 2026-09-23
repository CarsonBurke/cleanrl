# Predictive gradient control for standard clipped Beta PPO.
# Matched Adam / fixed SVRG / covariance-calibrated predictable control variate.
# The optimizer-only reference never participates in policy inference.
# SVRG: Johnson & Zhang (2013), Accelerating Stochastic Gradient Descent
# using Predictive Variance Reduction. Adaptive covariance calibration here
# estimates gradient predictiveness, not Bayesian parameter uncertainty.
import copy
import hashlib
import json
import os
import signal
import sys
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

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
from cleanrl.shared.host_actor import LeakyReluSq, ReluSq, SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


class PredictiveGradientControl:
    """Prequential covariance-calibrated control variate, not a weight posterior.

    The finite rollout is partitioned once. Uniform independent component draws
    make E[g_ref - mean_ref]=0 conditional on all past optimizer state. Therefore
    any coefficient chosen BEFORE the current draw preserves the gradient mean.
    Covariance calibration may be noisy/stale without invalidating that identity.
    """

    def __init__(self, template, mode="adaptive"):
        if mode not in ("adaptive", "fixed"):
            raise ValueError("mode must be adaptive or fixed")
        self.mode = mode
        self.mean = torch.zeros_like(template, dtype=torch.float64)
        self.variance = torch.zeros_like(self.mean)
        self.cross_sum = torch.zeros_like(self.mean)
        self.mean_innovation = torch.zeros_like(self.mean)
        self.reference_count = torch.zeros((), device=template.device, dtype=torch.float64)
        self.pair_count = torch.zeros_like(self.reference_count)

    @torch.no_grad()
    def reset(self):
        for value in (self.mean, self.variance, self.cross_sum, self.mean_innovation,
                      self.reference_count, self.pair_count):
            value.zero_()

    @torch.no_grad()
    def accumulate_reference(self, gradient):
        value = gradient.double()
        count = self.reference_count + 1
        difference = value - self.mean
        mean = self.mean + difference / count
        self.variance.add_(difference * (value - mean))
        self.mean.copy_(mean)
        self.reference_count.copy_(count)

    @torch.no_grad()
    def finish_reference(self):
        # The trainer supplies every nonempty, equal-size component exactly once.
        self.variance.div_(self.reference_count)

    def coefficients(self):
        if self.mode == "fixed":
            return torch.ones_like(self.mean)
        positive = self.variance > 0
        denominator = self.pair_count.clamp_min(1) * torch.where(positive, self.variance, 1.)
        # Cov(g_current, g_ref) = Var(g_ref) + Cov(g_current-g_ref, g_ref).
        # Projection solves the constrained variance-minimizing convex mixture;
        # these bounds are not evidence thresholds or learned learning rates.
        coefficient = (1 + self.cross_sum / denominator).clamp(0, 1)
        return torch.where(positive, coefficient, 0.)

    def correct(self, gradient, reference_gradient):
        coefficient = self.coefficients()
        centered = reference_gradient.double() - self.mean
        return (gradient.double() - coefficient * centered).to(gradient.dtype), coefficient

    @torch.no_grad()
    def step(self, gradient, reference_gradient):
        corrected, coefficient = self.correct(gradient, reference_gradient)
        innovation = gradient.double() - reference_gradient.double()
        centered = reference_gradient.double() - self.mean
        difference = innovation - self.mean_innovation
        # The intercept uses only past observations. It reduces calibration
        # variance without changing E[difference * centered] under iid draws.
        self.cross_sum.add_(difference * centered)
        self.pair_count.add_(1)
        self.mean_innovation.add_(difference / self.pair_count)
        return corrected, coefficient


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
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    gradient_mode: str = "adaptive"
    """matched iid-component PPO: adam, fixed, or adaptive gradient control"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    activation: str = "relu"
    """hidden activation: relu, relusq, leakyrelusq, or situglu"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 1024
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
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads"""
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


def dense_mlp(observation_dim, output_dim, activation, output_std):
    if activation == "situglu":
        modules = [
            init_situglu_branch(SiTUGLUBranch(observation_dim, 64)),
            init_situglu_branch(SiTUGLUBranch(64, 64)),
        ]
    else:
        activation_types = {
            "relu": nn.ReLU,
            "relusq": ReluSq,
            "leakyrelusq": LeakyReluSq,
        }
        try:
            activation_type = activation_types[activation]
        except KeyError as error:
            raise ValueError(f"unknown activation: {activation}") from error
        modules = [
            layer_init(nn.Linear(observation_dim, 64)), activation_type(),
            layer_init(nn.Linear(64, 64)), activation_type(),
        ]
    modules.append(layer_init(nn.Linear(64, output_dim), std=output_std))
    return nn.Sequential(*modules)


class Agent(nn.Module):
    def __init__(self, envs, args=None):
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
        activation = "relu" if args is None else args.activation
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive action range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = dense_mlp(observation_dim, 1, activation, 1.0)
        self.actor = dense_mlp(observation_dim, 2 * self.action_dim, activation, 0.01)

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
    if args.activation not in {"relu", "relusq", "leakyrelusq", "situglu"}:
        raise ValueError("activation must be relu, relusq, leakyrelusq or situglu")
    if args.gradient_mode not in {"adam", "fixed", "adaptive"}:
        raise ValueError("gradient_mode must be adam, fixed, or adaptive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("the finite objective requires equal-size components")
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0 or (args.norm_adv and args.minibatch_size < 2):
        raise ValueError("advantage normalization requires at least two samples per component")
    if args.target_kl is not None:
        raise ValueError("target_kl stopping is incompatible with the matched fixed update budget")
    if not args.cuda:
        raise ValueError("the predictive-gradient PPO trainer requires CUDA")
    return args


def loss_and_flat_gradient(model, loss_model, batch, compiled=True):
    """Own both snapshots outside compiled graphs before any next graph step.

    AOTAutograd compiles the backward of the compiled loss. Flattening inside that
    graph would return graph-pool storage which the reference call can overwrite.
    """
    if compiled:
        torch.compiler.cudagraph_mark_step_begin()
    model.zero_grad(set_to_none=True)
    loss, metrics = loss_model(*batch)
    loss.backward()
    gradient = torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
    return gradient, metrics.detach().clone()


@torch.no_grad()
def assign_flat_gradient(model, flat):
    """Install the corrected gradient in the actual PPO parameter grad leaves."""
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
        offset += count


def component_batch(fields, indices):
    return tuple(field.index_select(0, indices) for field in fields)


def gradient_audit(agent, reference, loss_model, reference_loss, fields, components,
                   controller, compiled):
    """Exact finite-objective moments at frozen parameters and prechosen lambda.

    No optimizer step or controller mutation occurs here. These temporary tables
    are diagnostics only and are never retained by the learner.
    """
    coefficient = controller.coefficients().detach().clone()
    reference_mean = controller.mean.detach().clone()
    raw = torch.empty((len(components), coefficient.numel()), device=coefficient.device,
                      dtype=torch.float64)
    frozen = torch.empty_like(raw)
    for index in range(len(components)):
        minibatch = component_batch(fields, components[index])
        gradient, _ = loss_and_flat_gradient(agent, loss_model, minibatch, compiled)
        raw[index].copy_(gradient)
        reference_gradient, _ = loss_and_flat_gradient(reference, reference_loss, minibatch, compiled)
        frozen[index].copy_(reference_gradient)
    # This reproduces the learner's FP32 corrected gradient, not just its ideal
    # real-arithmetic counterpart. Both means below are independently measured.
    corrected = (raw - coefficient * (frozen - reference_mean)).to(gradient.dtype).double()
    fixed = (raw - (frozen - reference_mean)).to(gradient.dtype).double()
    raw_mean, corrected_mean = raw.mean(0), corrected.mean(0)
    measured_reference_mean = frozen.mean(0)
    reference_centered = frozen - measured_reference_mean
    raw_centered = raw - raw_mean
    reference_variance = reference_centered.square().mean(0)
    covariance = (raw_centered * reference_centered).mean(0)
    positive = reference_variance > 0
    optimal = torch.where(positive, covariance / torch.where(positive, reference_variance, 1.), 0.).clamp(0, 1)
    raw_variance = raw_centered.square().mean(0)
    corrected_variance = (corrected - corrected_mean).square().mean(0)
    fixed_variance = (fixed - fixed.mean(0)).square().mean(0)
    optimal_variance = (raw_centered - optimal * reference_centered).square().mean(0)
    slices = {}
    offset = 0
    for name, parameter in agent.named_parameters():
        group = name.split(".", 1)[0]
        start, _ = slices.get(group, (offset, offset))
        offset += parameter.numel()
        slices[group] = (start, offset)
    metrics = {}
    for group, (start, stop) in slices.items():
        region = slice(start, stop)
        variance = raw_variance[region].sum()
        corrected_trace = corrected_variance[region].sum()
        fixed_trace = fixed_variance[region].sum()
        error = corrected_mean[region] - raw_mean[region]
        prefix = f"audit/{group}"
        metrics.update({
            f"{prefix}/raw_variance": variance,
            f"{prefix}/corrected_variance": corrected_trace,
            f"{prefix}/fixed_variance": fixed_trace,
            f"{prefix}/fixed_variance_ratio": torch.where(variance > 0, fixed_trace / variance, torch.nan),
            f"{prefix}/learned_to_fixed_variance_ratio": torch.where(fixed_trace > 0, corrected_trace / fixed_trace, torch.nan),
            f"{prefix}/variance_ratio": torch.where(variance > 0, corrected_trace / variance, torch.nan),
            f"{prefix}/optimal_variance_ratio": torch.where(variance > 0, optimal_variance[region].sum() / variance, torch.nan),
            f"{prefix}/mean_identity_l2": error.norm(),
            f"{prefix}/mean_identity_max_abs": error.abs().max(),
            f"{prefix}/mean_identity_relative_l2": error.norm() / raw_mean[region].norm().clamp_min(torch.finfo(torch.float64).tiny),
            f"{prefix}/coefficient_mean": coefficient[region].mean(),
            f"{prefix}/optimal_coefficient_mean": optimal[region].mean(),
            f"{prefix}/coefficient_optimal_mae": (coefficient[region] - optimal[region]).abs().mean(),
            f"{prefix}/reference_mean_error_l2": (measured_reference_mean[region] - reference_mean[region]).norm(),
        })
    return gather_metrics(metrics)


def json_safe(value):
    """JSON null means undefined/nonfinite, never a fabricated diagnostic zero."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root / "cleanrl" / "shared").glob("*.py")),
             *sorted((root / "cleanrl" / "shared").glob("*.c"))]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    registration = root / "runs" / "predictive_gradient_v5_preregistration.json"
    return dict(source_sha256=hashes, registration_path=str(registration.relative_to(root)),
                registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest() if registration.exists() else None,
                python=sys.version, torch=torch.__version__, numpy=np.__version__, gymnasium=gym.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())


def normalization_state(normalizer):
    fields = ("means", "variances", "counts", "returns", "gamma", "epsilon", "clip")
    return {name: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
            for name in fields if (value := getattr(normalizer, name, None)) is not None}


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
    run_name = f"{args.env_id}__{args.exp_name}_{args.gradient_mode}__{args.seed}__{time.time():.6f}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    run_dir = Path("runs") / run_name
    writer = SummaryWriter(str(run_dir))
    resources = ExitStack()
    resources.callback(writer.close)
    provenance = source_provenance()
    progress, episodes, audits = [], [], []
    global_step = warmup_transitions = completed_iterations = optimizer_steps = 0
    trained_transitions = 0
    agent = optimizer = obs_norm = rew_norm = controller = None
    stop_requested = False
    received_signal = None
    failure = None
    timer = PhaseTimer()
    phase_totals = {}
    start_time = time.perf_counter()

    def request_stop(signum, frame):
        nonlocal stop_requested, received_signal
        stop_requested, received_signal = True, signum

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous = signal.signal(signum, request_stop)
        resources.callback(signal.signal, signum, previous)

    def report(status):
        recent = [episode["return"] for episode in episodes[-100:]]
        return dict(
            args=vars(args), status=status, completed=status == "completed", run_name=run_name,
            provenance=provenance, requested_transitions=args.total_timesteps, actual_transitions=global_step,
            transitions=global_step, warmup_transitions=warmup_transitions,
            trained_transitions=trained_transitions, completed_rollouts=completed_iterations,
            planned_rollouts=args.num_iterations, optimizer_steps=optimizer_steps,
            unused_transition_budget=args.total_timesteps - global_step,
            elapsed_s=time.perf_counter() - start_time, phase_timings=phase_totals,
            final_return_mean_100=float(np.mean(recent)) if recent else None,
            final_returns=recent, episodic_returns=episodes, audits=audits, progress=progress,
            signal=received_signal, error=failure, checkpoint=f"{args.exp_name}.cleanrl_model"
            if agent is not None and (args.save_model or status != "completed") else None,
            fresh_initialization=True, checkpoint_loaded=False,
            gradient_protocol="fixed equal-size finite components; independent uniform draws with replacement",
            gradient_mode=args.gradient_mode, extra_policy_inference_networks=0,
            controller_state_bytes=0 if controller is None else sum(
                value.numel() * value.element_size() for value in vars(controller).values()
                if isinstance(value, torch.Tensor)),
            reference_parameter_bytes=0 if controller is None else sum(p.numel() * p.element_size() for p in agent.parameters()),
            limitations="Single-seed on-policy training returns, not held-out evaluation. Exact conditional finite-PPO-objective gradient-mean identity precedes nonlinear clipping and Adam; this is not an unbiased true policy-gradient claim. Covariance calibration may become stale within a rollout; rollout refresh uses the known data boundary, not unknown-changepoint detection. Full-rollout budget leaves a reported unused tail. Optimizer reference adds gradient evaluations and FP64 state; no claim of free LLM full-corpus references.",
        )

    try:
        atomic_json(run_dir / "progress.json", report("initializing"))
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        writer.add_text(
            "policy",
            f"Beta PPO; dense width-64 depth-2 {args.activation} trunk; FP32; "
            "native-action storage; host actor mirror; torch.compile",
        )
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        reference = copy.deepcopy(agent) if args.gradient_mode != "adam" else None
        if reference is not None:
            template = torch.empty(sum(p.numel() for p in agent.parameters()), device=device)
            controller = PredictiveGradientControl(template, mode=args.gradient_mode)
            del template
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        if reference is not None:
            def reference_loss(observations, native, old_logprobs, advantages, returns, old_values):
                return ppo_loss(reference, observations, native, old_logprobs, advantages, returns, old_values, args)

            reset_control = controller.reset
            accumulate_reference = controller.accumulate_reference
            finish_reference = controller.finish_reference
            control_step = controller.step
            if args.compile:
                # Mutable cross-call FP64 calibration state must not be graph-pool
                # owned. It remains compiled; only CUDA graph replay is disabled.
                options = {"triton.cudagraphs": False}
                reset_control = torch.compile(reset_control, fullgraph=True, options=options)
                accumulate_reference = torch.compile(accumulate_reference, fullgraph=True, options=options)
                finish_reference = torch.compile(finish_reference, fullgraph=True, options=options)
                control_step = torch.compile(control_step, fullgraph=True, options=options)
                reference_loss = torch.compile(reference_loss, mode=args.compile_mode, fullgraph=True, dynamic=False)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
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
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        component_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        max_updates = args.update_epochs * args.num_minibatches
        update_metrics = torch.empty((max_updates, 6), device=device)
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
        warmup_transitions = global_step
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        audit_thresholds = (500000, 1000000, 2000000, 4000000)
        audited_thresholds = set()
        for iteration in range(1, args.num_iterations + 1):
            if stop_requested:
                break
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                if stop_requested:
                    break
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
                        episodes.append({"step": global_step, "return": episode_return,
                                         "length": float(info["episode"]["l"])})
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            if stop_requested:
                break
            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs = rollout_statistics(b_obs, b_native)
                b_values, b_logprobs = b_values.clone(), b_logprobs.clone()
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
            fields = (b_obs, b_native, b_logprobs, b_advantages, b_returns, b_values)
            # Membership and minibatch-local advantage normalization stay fixed
            # throughout this rollout's finite objective, including references.
            components = torch.stack(device_minibatches(
                args.batch_size, args.minibatch_size, device, shuffle_generator))
            draws = torch.randint(args.num_minibatches, (max_updates, 1), device=device,
                                  generator=component_generator)
            if reference is not None:
                with timer.span("reference"):
                    reference.load_state_dict(agent.state_dict())
                    reset_control()
                    for component in range(args.num_minibatches):
                        if stop_requested:
                            break
                        reference_gradient, _ = loss_and_flat_gradient(
                            reference, reference_loss, component_batch(fields, components[component]), args.compile)
                        accumulate_reference(reference_gradient)
                    if not stop_requested:
                        finish_reference()
                    reference.zero_grad(set_to_none=True)
            if stop_requested:
                break
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for draw in range(epoch * args.num_minibatches, (epoch + 1) * args.num_minibatches):
                        if stop_requested:
                            break
                        # A CUDA scalar as a Python subscript would silently
                        # synchronize. index_select consumes a device vector.
                        indices = components.index_select(0, draws[draw]).flatten()
                        minibatch = component_batch(fields, indices)
                        if controller is None:
                            if args.compile:
                                torch.compiler.cudagraph_mark_step_begin()
                            optimizer.zero_grad(set_to_none=True)
                            loss, metrics = loss_model(*minibatch)
                            loss.backward()
                            update_metrics[updates].copy_(metrics)
                        else:
                            gradient, metrics = loss_and_flat_gradient(agent, loss_model, minibatch, args.compile)
                            # The helper owns both snapshots before the reference
                            # call begins a new CUDA-graph step.
                            update_metrics[updates].copy_(metrics)
                            reference_gradient, _ = loss_and_flat_gradient(reference, reference_loss, minibatch, args.compile)
                            corrected, _ = control_step(gradient, reference_gradient)
                            assign_flat_gradient(agent, corrected)
                            reference.zero_grad(set_to_none=True)
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer_steps += 1
                        updates += 1
                    if stop_requested:
                        break
            if stop_requested:
                break
            completed_iterations += 1
            trained_transitions += args.batch_size
            audit_reasons = [str(threshold) for threshold in audit_thresholds
                             if global_step >= threshold and threshold not in audited_thresholds]
            if iteration == 1:
                audit_reasons.append("first")
            if iteration == args.num_iterations:
                audit_reasons.append("final")
            if controller is not None and audit_reasons:
                with timer.span("audit"):
                    audit = gradient_audit(agent, reference, loss_model, reference_loss,
                                           fields, components, controller, args.compile)
                audits.append(dict(step=global_step, rollout=iteration, reasons=audit_reasons, **audit))
                audited_thresholds.update(threshold for threshold in audit_thresholds if global_step >= threshold)
                for name, value in audit.items():
                    writer.add_scalar(name, value, global_step)

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
            })
            if controller is not None:
                coefficient = controller.coefficients()
                logged.update(gather_metrics({
                    "control/coefficient_mean": coefficient.mean(),
                    "control/coefficient_min": coefficient.min(),
                    "control/coefficient_max": coefficient.max(),
                    "control/coefficient_zero_fraction": (coefficient == 0).double().mean(),
                    "control/coefficient_one_fraction": (coefficient == 1).double().mean(),
                    "control/reference_variance_trace": controller.variance.sum(),
                    "control/calibration_pairs": controller.pair_count,
                    "control/reference_components": controller.reference_count,
                }))
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            timings = timer.summary()
            for phase, timing in timings.items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
                phase_totals[phase] = phase_totals.get(phase, 0.) + timing["total_s"]
            progress.append(dict(step=global_step, iteration=iteration, updates=updates,
                                 wall_s=now - start_time, timings=timings, **logged))
            atomic_json(run_dir / "progress.json", report("running"))
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        status = ("failed" if failure is not None else "cancelled" if stop_requested else
                  "completed" if completed_iterations == args.num_iterations else "incomplete")
        try:
            if agent is not None and (args.save_model or status != "completed"):
                checkpoint = dict(agent=agent.state_dict(), args=vars(args),
                                  observation_normalization=normalization_state(obs_norm) if obs_norm is not None else None,
                                  reward_normalization=normalization_state(rew_norm) if rew_norm is not None else None,
                                  actual_transitions=global_step, completed_rollouts=completed_iterations,
                                  status=status, provenance=provenance)
                temporary = run_dir / f"{args.exp_name}.cleanrl_model.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(run_dir / f"{args.exp_name}.cleanrl_model")
            final_report = report(status)
            atomic_json(run_dir / "progress.json", final_report)
            atomic_json(run_dir / "result.json", final_report)
            print("TRAINING_RESULT=" + json.dumps(json_safe({key: value for key, value in final_report.items()
                  if key not in {"args", "progress", "episodic_returns", "audits", "provenance"}})), flush=True)
        finally:
            resources.close()


if __name__ == "__main__":
    main()
