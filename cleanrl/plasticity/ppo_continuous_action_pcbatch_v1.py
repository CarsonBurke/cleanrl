"""Per-perceptron, per-sample plasticity INSIDE the PPO minibatch gradient sum.

# WHY THIS FILE EXISTS

Every diagnostic in `cleanrl/plasticity/` so far ran the per-perceptron rules at
batch size 1, where the mechanism is provably degenerate: one sample gives
`row_h = d_h * x`, so every unit's update is parallel to the same `x` and a
per-unit weight can only be a scalar step-size change. `pc_stream.py` escaped
that only by masking receptive fields (`--field-fraction 0.25`); at full fan-in
and batch 1 its own report measures the state->unit correspondence as worth
EXACTLY ZERO.

PPO does not run at batch 1. Its weight gradient for a hidden site is

    row_h = sum_t d_th * x_t                       (t indexes the minibatch)

so a weight `c_th` that depends on the unit AND the sample gives

    row_h = sum_t c_th * d_th * x_t

and each unit's update is steered to a DIFFERENT direction inside the span of
the minibatch inputs. That is a genuine per-unit reallocation, unreachable by
any per-unit scalar, and it is the regime this file implements.

# MECHANISM

Each hidden unit `h` carries a small local generative model of its OWN incoming
pre-activation stream, and nothing else:

    z_th   = (pre_th - mean_h) / sd_h              unit h's own state, self-standardised
                                                   by its own cumulative moments
    phi_th = normalise(rbf(z_th))                  K soft bins over that unit's own state axis
    e_th   = |phi_th - U_h U_h^T phi_th|^2         what unit h's own rank-R model does not
                                                   anticipate about this sample
    c_th   = e_th / s_h,   s_h = cumulative mean of e_th for unit h

`U_h` is learned online by Oja's subspace rule on the unit's own prediction
error and receives NO gradient from the PPO loss. `c_th` is applied inside the
minibatch sum by scaling `d_th` before the outer product, so the layer's forward
is bit-identical to `nn.Linear` and only the weight gradient changes.

Traps this file refuses, all of which already cost results in this family:
  - the reference `s_h` is the unit's OWN cumulative mean of its OWN error. No
    global statistic, no cross-unit normalisation, and no mean-one budget across
    units -- a budget makes a rule exactly blind to global shifts, and that
    faked four earlier results here.
  - no learning-rate confound by construction: `c` is a ratio to the unit's own
    running mean, so its realised mean sits at 1 and is LOGGED every interval.
    If `plasticity/c_mean` drifts from 1 the arm is an LR change in disguise and
    must be reported as one.
  - `c` is state-dependent per SAMPLE. `plasticity/c_sample_dispersion` is the
    number that separates this from the degenerate per-unit scalar; if it is
    ~0 the run is `--pc-scalar` with extra steps.
  - the gate never touches an output layer: the Beta head and the value head are
    plain `nn.Linear`. A gate read off a head whose weights start near zero
    deadlocks it.

# ARMS

  default        per-(sample, unit) `c` on the four hidden sites
  --pc-scalar    `c` averaged over the minibatch, i.e. per-unit only -- the
                 degenerate case the batch-1 diagnostics were stuck in
  --pc-shared    ONE model for the whole layer (identical init, updates averaged
                 over units), so per-unit content is removed but the level stays
  --pc-shuffle   `c` permuted across units inside the minibatch. Both marginals
                 (per-unit and per-sample) are preserved exactly; only the
                 state->unit correspondence dies. If this ties the default arm,
                 state dependence contributed nothing.
  --pc-off       mechanism inert; bit-identical to `ppo_continuous_action.py`

# HYPOTHESIS

Per-(sample, unit) reallocation of the minibatch gradient toward the samples a
unit's own model has not yet explained keeps units specialised on distinct parts
of the state distribution, which is exactly the failure mode of PPO's shared
64-unit trunk under a shifting on-policy distribution. Expected to show up as
retention of earlier behaviour rather than faster acquisition, per pc_stream.
"""
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
PLASTICITY_METRICS = ("c_mean", "c_unit_dispersion", "c_sample_dispersion", "model_loss")

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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

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
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Per-perceptron, per-sample plasticity. Defaults are the live arm.
    pc_rank: int = 2
    """rank of each unit's local generative model of its own pre-activation"""
    pc_features: int = 8
    """soft bins spanning the unit's own standardized pre-activation axis"""
    pc_lr: float = 0.02
    """Oja rate for the local models; they get no gradient from the PPO loss"""
    pc_init: float = 0.05
    """initial scale of the local models; near zero means inert at init"""
    pc_cap: float = 5.0
    """upper clamp on c, guarding a unit whose own error reference collapses"""
    pc_off: bool = False
    """mechanism inert: bit-identical to ppo_continuous_action.py"""
    pc_scalar: bool = False
    """ablation: c constant across samples (per-unit only, the degenerate case)"""
    pc_shared: bool = False
    """ablation: one local model shared by every unit in the layer"""
    pc_shuffle: bool = False
    """ablation: permute the state->unit correspondence, preserving both marginals"""

    # Execution controls, independent of PPO's batch and optimizer settings.
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


# Public evaluation and historical GAE compatibility helpers; not the training path.
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
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


@dataclass(frozen=True)
class PlasticConfig:
    """Everything the local models need; one instance is shared by all sites."""
    rank: int = 2
    features: int = 8
    lr: float = 0.02
    init: float = 0.05
    cap: float = 5.0
    active: bool = True
    scalar: bool = False
    shared: bool = False
    shuffle: bool = False
    seed: int = 1


class _GateGradient(torch.autograd.Function):
    """Identity forward, per-(sample, unit) scaled backward.

    The pre-activation tensor is returned untouched, so the layer's value is the
    `nn.Linear` value bit for bit. `d_th` is scaled on the way back, which puts
    the weight gradient at `sum_t c_th * d_th * x_t` exactly -- the scaling
    happens inside the minibatch sum, not on the assembled row. The same scaled
    delta propagates to earlier layers, which is the honest consequence of a
    local rule acting on a unit's own error signal.
    """

    @staticmethod
    def forward(ctx, pre, weight):
        ctx.save_for_backward(weight)
        return pre.view_as(pre)

    @staticmethod
    def backward(ctx, grad_pre):
        (weight,) = ctx.saved_tensors
        return grad_pre * weight, None


class PlasticLinear(nn.Linear):
    """`nn.Linear` whose weight gradient is `sum_t c_th * d_th * x_t`.

    Subclasses `nn.Linear` deliberately: the host rollout mirror, `layer_init`
    and the checkpoint layout all keep working, and the forward path is the
    unmodified `F.linear`. The local models live in buffers, so the optimizer
    and `clip_grad_norm_` never see them and the PPO loss cannot reach them.
    """

    def __init__(self, in_features, out_features, config=None, generator=None):
        super().__init__(in_features, out_features)
        self.config = config
        self.active = config is not None and config.active
        if config is None:
            return
        bins, rank = config.features, config.rank
        # A dedicated generator: the global stream must be consumed exactly as
        # the baseline consumes it, or every weight downstream of here differs.
        model = torch.randn((out_features, bins, rank), generator=generator)
        model.mul_(config.init / np.sqrt(bins))
        if config.shared:
            model = model[:1].expand(out_features, bins, rank).contiguous()
        self.register_buffer("model", model)
        self.register_buffer("centres", torch.linspace(-2.0, 2.0, bins))
        self.register_buffer("pre_sum", torch.zeros(out_features))
        self.register_buffer("pre_sq_sum", torch.zeros(out_features))
        self.register_buffer("err_sum", torch.zeros(out_features))
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("stats", torch.zeros(len(PLASTICITY_METRICS)))
        self.register_buffer("perm", torch.arange(out_features))
        self.bin_width = 4.0 / max(bins - 1, 1)

    def forward(self, x):
        pre = F.linear(x, self.weight, self.bias)
        # Rollout statistics, GAE values and truncation bootstraps run under
        # no_grad and must not move the local models: only the optimizer path
        # carries a delta for the plasticity to act on.
        if not self.active or not torch.is_grad_enabled():
            return pre
        return _GateGradient.apply(pre, self._plasticity(pre.detach()))

    @torch.no_grad()
    def _plasticity(self, pre):
        """Per-(sample, unit) weight from each unit's own pre-activation."""
        rows = float(pre.shape[0])
        # Cumulative moments, not EMAs: sufficient statistics of the unit's own
        # state stream with no forgetting horizon to tune.
        self.count.add_(rows)
        self.pre_sum.add_(pre.sum(0))
        self.pre_sq_sum.add_(pre.square().sum(0))
        mean = self.pre_sum / self.count
        variance = (self.pre_sq_sum / self.count - mean.square()).clamp_min(1e-8)
        state = (pre - mean) * variance.rsqrt()                       # (B, H)
        feature = torch.exp(-0.5 * ((state.unsqueeze(-1) - self.centres)
                                    / self.bin_width).square())       # (B, H, K)
        feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        latent = torch.einsum("bhk,hkr->bhr", feature, self.model)
        residual = feature - torch.einsum("bhr,hkr->bhk", latent, self.model)
        error = residual.square().sum(-1)                             # (B, H) in [0, 1]
        # The reference is this unit's own cumulative mean of its own error,
        # over strictly EARLIER samples. Normalising by the current minibatch
        # would be a within-batch budget, which is the trap.
        seen = self.count - rows
        reference = (self.err_sum / seen.clamp_min(1.0)).clamp_min(1e-8)
        weight = (error / reference).clamp(0.0, self.config.cap)
        weight = torch.where(seen > 0.0, weight, torch.ones_like(weight))
        if self.config.scalar:
            # The degenerate per-unit case: identical realized mean, sample
            # dependence destroyed.
            weight = weight.mean(0, keepdim=True).expand_as(weight)
        if self.config.shuffle:
            # Both marginals survive a pure relabelling of the unit axis; only
            # the state->unit correspondence dies.
            weight = weight.index_select(1, self.perm)
        self.err_sum.add_(error.sum(0))
        # Oja's subspace rule on the unit's OWN prediction error. No PPO
        # gradient reaches this; the model only minimises its own objective.
        update = torch.einsum("bhk,bhr->hkr", residual, latent) / rows
        if self.config.shared:
            update = update.mean(0, keepdim=True).expand_as(update)
        self.model.add_(update, alpha=self.config.lr)
        norm = self.model.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.model.mul_(norm.clamp_max(1.0) / norm)
        self.stats.copy_(torch.stack((
            weight.mean(),                                # is this an LR change?
            weight.mean(0).std(correction=0),             # dispersion across units
            weight.std(0, correction=0).mean(),           # dispersion within a unit
            error.mean(),                                 # the local models' own loss
        )))
        return weight


class Agent(nn.Module):
    def __init__(self, envs, plastic=None):
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
        generator = None if plastic is None else torch.Generator().manual_seed(plastic.seed + 8191)

        def hidden(in_features, out_features):
            return layer_init(PlasticLinear(in_features, out_features, plastic, generator))

        # Hidden sites only. The value head and the Beta head stay plain: a gate
        # read off a head initialised near zero pins it there forever.
        self.critic = nn.Sequential(
            hidden(observation_dim, 64), nn.Tanh(),
            hidden(64, 64), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            hidden(observation_dim, 64), nn.Tanh(),
            hidden(64, 64), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

    def plastic_sites(self):
        return [module for module in self.modules() if isinstance(module, PlasticLinear)]

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


def plastic_config(args):
    """The plasticity settings for this run; buffers exist even when inert."""
    return PlasticConfig(
        rank=args.pc_rank, features=args.pc_features, lr=args.pc_lr,
        init=args.pc_init, cap=args.pc_cap, active=not args.pc_off,
        scalar=args.pc_scalar, shared=args.pc_shared, shuffle=args.pc_shuffle,
        seed=args.seed,
    )


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if args.pc_features < 2 or args.pc_rank < 1 or args.pc_rank >= args.pc_features:
        raise ValueError("the local model needs 1 <= pc_rank < pc_features and pc_features >= 2")
    if args.pc_lr < 0.0 or args.pc_init < 0.0 or args.pc_cap < 1.0:
        raise ValueError("pc_lr and pc_init must be non-negative and pc_cap at least 1")
    ablations = sum((args.pc_scalar, args.pc_shared, args.pc_shuffle))
    if ablations > 1:
        raise ValueError("pc_scalar, pc_shared and pc_shuffle are distinct control arms")
    if args.pc_off and ablations:
        raise ValueError("pc_off is inert; it cannot be combined with an ablation")
    if args.pc_scalar and args.minibatch_size < 2:
        raise ValueError("pc_scalar is only defined for minibatches of at least two samples")
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
        arm = "off" if args.pc_off else next(
            (name for name, on in (("scalar", args.pc_scalar), ("shared", args.pc_shared),
                                   ("shuffle", args.pc_shuffle)) if on), "pc")
        writer.add_text("plasticity", f"arm={arm}; c_th=e_th/s_h from each unit's own preactivation, "
                                      f"applied inside the minibatch gradient sum")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, plastic_config(args)).to(device)
        sites = agent.plastic_sites()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout. The
        # gate is gradient-only, so the mirror stays a plain MLP.
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
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        # Nor may the shuffle control: its permutations are refreshed per
        # minibatch, outside the compiled loss, into a static buffer.
        permute_generator = torch.Generator(device=device).manual_seed(args.seed + 77)
        permuting = args.pc_shuffle and not args.pc_off

        def refresh_permutations():
            for site in sites:
                site.perm.copy_(torch.randperm(site.out_features, device=device,
                                               generator=permute_generator))

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
                        if permuting:
                            refresh_permutations()
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            # The plasticity numbers come from the last minibatch's applied c,
            # averaged over sites. c_mean away from 1 means the arm is a
            # step-size change; c_sample_dispersion at 0 means it collapsed to
            # the degenerate per-unit scalar.
            plasticity = torch.stack([site.stats for site in sites]).mean(0)
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                **{f"plasticity/{name}": plasticity[index]
                   for index, name in enumerate(PLASTICITY_METRICS)},
            })
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
            from cleanrl_utils.evals.ppo_eval import evaluate
            episodic_returns = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10,
                run_name=f"{run_name}-eval", Model=Agent, device=device, gamma=args.gamma,
            )
            for index, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, index)
            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
