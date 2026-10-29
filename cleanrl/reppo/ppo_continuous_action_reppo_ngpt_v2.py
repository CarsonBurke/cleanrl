# REPPO v2: REPPO (reppo_v1) with the architecture and optimizer lessons of the PPO base
# (awr/ppo_continuous_action_ppo_sepclip_control_v1.py, residual_stiglu_ngpt_scaled 50M):
# - Both networks use the residual SiTU-GLU trunk: unit-L2 residual directions, sqrt(width)
#   branch/readout scale, nGPT unit-norm matrix rows restored after every optimizer step, learned
#   readout gains. The critic's features (the Q encoder output, also the self-predictive target)
#   therefore live on a fixed-radius sphere; the next-feature predictor is a residual step on it.
# - Adam(eps=1e-5) with a linearly annealed learning rate on the networks (a device tensor, so the
#   captured update graphs see the schedule); the temperature and KL multiplier keep REPPO's
#   AdamW rate in their own group.
# - Each network is clipped to its own max_grad_norm budget; the two dual variables are not
#   clipped with the actor (REPPO clips them jointly).
# Hypothesis: the trunk that made PPO's critic/actor stable at high learning rates also gives a
# better-conditioned Q(s, a) landscape for the pathwise actor gradient.
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
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.graph_step import capture_steps
from cleanrl.shared.hl_gauss import HLGaussConfig
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics, get_gae_fn
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
CRITIC_METRICS = ("critic_ce", "aux_loss", "q_mse")
ACTOR_METRICS = ("actor_objective", "q_pi", "entropy", "kl", "kl_violation", "temperature",
                 "kl_multiplier", "std")


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
    learning_rate: float = 0.0024
    """Adam learning rate of the actor and critic networks"""
    anneal_lr: bool = True
    """linearly anneal the network learning rate to zero"""
    dual_learning_rate: float = 3e-4
    """AdamW learning rate of the temperature and KL multiplier"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gve_lambda: float = 0.95
    """lambda of the soft lambda-return critic target"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs of critic and actor updates per rollout"""
    max_grad_norm: float = 0.5
    """per-network gradient clipping norm"""
    reward_scale: float = 0.1
    """fixed reward multiplier (keeps the categorical support stationary)"""

    # Critic
    critic_hidden_dim: int = 64
    """critic trunk width"""
    num_bins: int = 151
    """HL-Gauss categories"""
    v_min: float = -50.0
    """lowest supported scaled soft return"""
    v_max: float = 250.0
    """highest supported scaled soft return"""
    aux_loss_coef: float = 1.0
    """weight of the next-state feature prediction loss"""

    # Actor
    actor_hidden_dim: int = 64
    """actor trunk width"""
    actor_min_std: float = 0.0
    """additive floor on the pre-squash standard deviation"""
    kl_bound: float = 0.1
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
    ent_start: float = 0.01
    """initial entropy temperature"""
    ent_target_mult: float = 0.5
    """target entropy is -ent_target_mult * action_dim"""

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile the losses and capture both update steps as CUDA graphs"""
    compile_mode: str = "reduce-overhead"
    """accepted for launcher compatibility; every path here uses Inductor without cudagraph trees"""
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


def branch(in_dim, width):
    return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, width)))


class ResidualTrunk(nn.Module):
    """Two SiTU-GLU stages on the unit sphere; returns sqrt(width) * unit residual direction."""

    def __init__(self, in_dim, width):
        super().__init__()
        self.width = width
        self.scale = float(np.sqrt(width))
        self.first = branch(in_dim, width)
        self.second = branch(width, width)

    def forward(self, x):
        h = F.normalize(self.first(x), dim=-1)
        stage = F.normalize(self.second(self.scale * h), dim=-1)
        return self.scale * F.normalize(h + stage, dim=-1)


class Readout(nn.Module):
    """Unit-row linear readout with a learned per-output gain.

    Row normalization discards any init scale of ``linear``, so the output scale at init is set by
    ``gain``: unit rows against a sqrt(width)-norm input give O(gain) outputs.
    """

    def __init__(self, width, out_dim, gain=1.0):
        super().__init__()
        self.linear = layer_init(nn.Linear(width, out_dim), std=1.0)
        self.gain = nn.Parameter(torch.full((out_dim,), float(gain)))

    def forward(self, x):
        return self.gain * self.linear(x)


def normalize_rows(modules):
    """nGPT matrix axes (branch gate/up rows, down columns, readout rows); biases and gains untouched."""
    for module in modules:
        if isinstance(module, SiTUGLUBranch):
            pairs = ((module.gate.weight, 1), (module.up.weight, 1), (module.down.weight, 0))
        elif isinstance(module, Readout):
            pairs = ((module.linear.weight, 1),)
        else:
            continue
        for weight, dim in pairs:
            weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))


def squashed_sample(mean, std):
    """Reparameterized tanh-Gaussian sample and its exact log-density."""
    noise = torch.randn_like(mean)
    pre = mean + std * noise
    # log|d tanh(u)/du| = 2 (log 2 - u - softplus(-2u)), stable for large |u|.
    log_det = 2.0 * (LOG_2 - pre - F.softplus(-2.0 * pre))
    logprob = (-0.5 * noise.square() - std.log() - LOG_SQRT_2PI - log_det).sum(-1)
    return torch.tanh(pre), logprob


def gaussian_kl(old_mean, old_std, mean, std):
    """KL(old || new) of diagonal Gaussians, equal to the KL of their tanh pushforwards."""
    return (torch.log(std / old_std)
            + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square()) - 0.5).sum(-1)


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, args):
        super().__init__()
        self.trunk = ResidualTrunk(observation_dim, args.actor_hidden_dim)
        # Near-zero mean and log-std at init (std ~ 1 everywhere, as in REPPO), not O(1) random ones.
        self.head = Readout(args.actor_hidden_dim, 2 * action_dim, gain=0.01)
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(args.ent_start))))
        self.log_kl_multiplier = nn.Parameter(torch.tensor(float(np.log(args.kl_start))))
        self.min_std = args.actor_min_std

    def network_parameters(self):
        return [*self.trunk.parameters(), *self.head.parameters()]

    def forward(self, observations):
        mean, log_std = self.head(self.trunk(observations)).chunk(2, dim=-1)
        return mean, log_std.exp() + self.min_std


class Critic(nn.Module):
    """REPPO's Q(s, a) on the residual trunk: spherical features -> {categorical value, next features}."""

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        width = args.critic_hidden_dim
        self.encoder = ResidualTrunk(observation_dim + action_dim, width)
        self.value_head = Readout(width, args.num_bins)
        self.predictor = branch(width, width)
        self.hl = HLGaussConfig(v_min=args.v_min, v_max=args.v_max, num_bins=args.num_bins).build(device)
        self.zero_dist = nn.Parameter(self.hl.project(torch.zeros((), device=device)))

    def features(self, observations, actions):
        return self.encoder(torch.cat((observations, actions), dim=-1))

    def logits(self, features):
        return self.value_head(features) + ZERO_DIST_SCALE * self.zero_dist

    def predict_next(self, features):
        # A residual step on the feature sphere: the identity prior is "features barely move".
        unit = features / self.encoder.scale
        return self.encoder.scale * F.normalize(unit + F.normalize(self.predictor(features), dim=-1), dim=-1)

    def value(self, observations, actions):
        return self.hl.to_scalar(self.logits(self.features(observations, actions)))


class ResidualHostMirror:
    """Fused FP32 host mirror of ResidualTrunk + Readout, composed from per-stage mirrors.

    Stage outputs are permanent buffers; copy the first before calling the second, and add
    into our own permanent buffer before evaluating the head.
    """

    def __init__(self, actor, num_envs):
        width = actor.trunk.width
        self.first = make_host_mirror(actor.trunk.first, num_envs)
        self.second = make_host_mirror(actor.trunk.second, num_envs)
        self.head = make_host_mirror(nn.Sequential(actor.head.linear), num_envs)
        self.scale = np.float32(actor.trunk.scale)
        self.hidden = np.empty((num_envs, width), dtype=np.float32)
        self.branch = np.empty_like(self.hidden)
        self.squared = np.empty_like(self.hidden)
        self.norm = np.empty((num_envs, 1), dtype=np.float32)
        self.actor = actor
        self.gain = np.empty(actor.head.gain.numel(), dtype=np.float32)
        self.fused = all(m.fused for m in (self.first, self.second, self.head))
        self.fallback_reason = None
        self.refresh()

    def refresh(self):
        self.first.refresh()
        self.second.refresh()
        self.head.refresh()
        np.copyto(self.gain, self.actor.head.gain.detach().cpu().numpy())

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        # Match F.normalize's zero-vector definition, including its epsilon.
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    def __call__(self, observations):
        np.copyto(self.hidden, self.first(observations))
        self._normalize(self.hidden)
        np.multiply(self.hidden, self.scale, out=self.branch)
        np.copyto(self.branch, self.second(self.branch))
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, self.scale, out=self.hidden)
        output = self.head(self.hidden)
        np.multiply(output, self.gain, out=output)
        return output


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs."""

    def __init__(self, num_envs, action_dim, min_std):
        shape = (num_envs, action_dim)
        self.min_std = np.float32(min_std)
        self.std = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
        np.exp(log_std, out=self.std)
        self.std += self.min_std
        rng.standard_normal(self.action.shape, dtype=np.float32, out=self.action)
        self.action *= self.std
        self.action += mean
        np.tanh(self.action, out=self.action)
        return self.action


def soft_lambda_returns(gae_fn, rewards, next_values, terminations, truncations, gamma, lam):
    """REPPO's GVE target G_t = r_t + gamma [(1 - lam) Q'_t + lam G_{t+1}] via the shared GAE kernel.

    Q'_t = Q(s_{t+1}, a'), a' ~ pi. Boundaries (and the rollout end) bootstrap from Q'_t alone;
    terminations do not bootstrap. Explicit-next-value GAE returns A_t + v_t with
    A_t = r_t + gamma Q'_t - v_t + gamma lam A_{t+1}, which is exactly G_t when v_{t+1} = Q'_t:
    set v_t = Q'_{t-1} (v_0 cancels).
    """
    values = torch.cat((torch.zeros_like(next_values[:1]), next_values[:-1]))
    _, returns = gae_fn(rewards, values, terminations, truncations, next_values, gamma, lam)
    return returns


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_minibatches must divide the batch: update steps are fixed-shape graphs")
    args.minibatch_size = args.batch_size // args.num_minibatches
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
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        space = envs.single_action_space
        if not (isinstance(space, gym.spaces.Box) and np.all(space.low == -1.0) and np.all(space.high == 1.0)):
            raise ValueError("the tanh policy requires a [-1, 1] Box action space")
        obs_shape = envs.single_observation_space.shape
        observation_dim, action_dim = int(np.prod(obs_shape)), int(np.prod(space.shape))
        actor = Actor(observation_dim, action_dim, args).to(device)
        critic = Critic(observation_dim, action_dim, args, device).to(device)
        actor_modules, critic_modules = list(actor.modules()), list(critic.modules())
        with torch.no_grad():
            normalize_rows(actor_modules)
            normalize_rows(critic_modules)
        # A device learning rate: captured optimizer steps read it, so the schedule reaches them.
        network_lr = torch.tensor(args.learning_rate, device=device)
        actor_network, critic_parameters = actor.network_parameters(), list(critic.parameters())
        dual_parameters = [actor.log_temperature, actor.log_kl_multiplier]
        actor_optimizer = torch.optim.AdamW(
            [{"params": actor_network, "lr": network_lr, "weight_decay": 0.0, "eps": 1e-5},
             {"params": dual_parameters, "lr": args.dual_learning_rate}], fused=True)
        critic_optimizer = torch.optim.Adam(critic_parameters, lr=network_lr, eps=1e-5, fused=True)
        actor_parameters = actor_network + dual_parameters
        target_entropy = -args.ent_target_mult * action_dim
        B, M = args.batch_size, args.minibatch_size

        # Persistent device batch read by the captured update steps.
        f32 = dict(device=device, dtype=torch.float32)
        b_obs = torch.zeros((B, observation_dim), **f32)
        b_actions = torch.zeros((B, action_dim), **f32)
        b_labels = torch.zeros((B, args.num_bins), **f32)
        b_targets = torch.zeros(B, **f32)
        b_next_features = torch.zeros((B, args.critic_hidden_dim), **f32)
        b_continuing = torch.zeros(B, **f32)
        b_old_mean = torch.zeros((B, action_dim), **f32)
        b_old_std = torch.ones((B, action_dim), **f32)
        minibatch = torch.zeros(M, dtype=torch.long, device=device)
        critic_sums = torch.zeros(len(CRITIC_METRICS), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(2, **f32)

        def rollout_statistics(observations, actions, next_observations):
            """Behavior-policy moments, sampled next-state soft values and features, in one pass."""
            old_mean, old_std = actor(observations)
            next_mean, next_std = actor(next_observations)
            next_actions, next_logprob = squashed_sample(next_mean, next_std)
            next_features = critic.features(next_observations, next_actions)
            next_q = critic.hl.to_scalar(critic.logits(next_features))
            data_q = critic.value(observations, actions)
            return old_mean, old_std, next_logprob, next_q, next_features, data_q

        def critic_loss(obs, actions, labels, targets, next_features, continuing):
            features = critic.features(obs, actions)
            logits = critic.logits(features)
            ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            prediction_error = (critic.predict_next(features) - next_features).square().mean(-1)
            aux = (continuing * prediction_error).mean()
            q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
            return ce + args.aux_loss_coef * aux, torch.stack((ce.detach(), aux.detach(), q_mse))

        def actor_loss(obs, old_mean, old_std):
            mean, std = actor(obs)
            actions, logprob = squashed_sample(mean, std)
            q = critic.value(obs, actions)
            temperature = actor.log_temperature.exp()
            kl_multiplier = actor.log_kl_multiplier.exp()
            objective = -q + temperature.detach() * logprob
            kl = gaussian_kl(old_mean, old_std, mean, std)
            inside = kl < args.kl_bound
            # Outside the trust region a state only receives the pull back toward the old policy.
            policy_loss = torch.where(inside, objective, kl * kl_multiplier.detach()).mean()
            entropy = -logprob
            temperature_loss = temperature * (entropy - target_entropy).detach().mean()
            multiplier_loss = -kl_multiplier * (kl.mean() - args.kl_bound).detach()
            metrics = torch.stack((objective.mean(), q.mean(), entropy.mean(), kl.mean(),
                                   1.0 - inside.float().mean(), temperature, kl_multiplier,
                                   std.mean())).detach()
            return policy_loss + temperature_loss + multiplier_loss, metrics

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        @torch.no_grad()
        def restore_actor_rows():
            normalize_rows(actor_modules)

        @torch.no_grad()
        def restore_critic_rows():
            normalize_rows(critic_modules)

        def critic_step():
            idx = minibatch
            critic_optimizer.zero_grad(set_to_none=False)
            loss, metrics = critic_loss(b_obs[idx], b_actions[idx], b_labels[idx], b_targets[idx],
                                        b_next_features[idx], b_continuing[idx])
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            restore_critic_rows()
            critic_sums.add_(metrics)

        def actor_step():
            idx = minibatch
            actor_optimizer.zero_grad(set_to_none=False)
            loss, metrics = actor_loss(b_obs[idx], b_old_mean[idx], b_old_std[idx])
            # The critic is only differentiated through its action input.
            loss.backward(inputs=actor_parameters)
            grad_sums[0] += nn.utils.clip_grad_norm_(actor_network, args.max_grad_norm, foreach=True)
            actor_optimizer.step()
            restore_actor_rows()
            actor_sums.add_(metrics)

        update_steps = None  # captured lazily, once the first rollout populated the batch
        # The recurrence is one Triton kernel; a reduce-overhead (cudagraph-tree) wrapper would record
        # in the same process as the manually captured update graphs, which faults their replays.
        gae_fn = get_gae_fn(compiled=args.compile, mode="default", explicit_next_values=True)
        hl_project = critic.hl.project

        host_actor = ResidualHostMirror(actor, args.num_envs)
        writer.add_text("host_actor", f"fused={host_actor.fused} fallback={host_actor.fallback_reason}")
        sampler = np.random.default_rng(args.seed)
        sample_actions = HostSquashedGaussian(args.num_envs, action_dim, args.actor_min_std)

        def act(observations):
            action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return action

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   store_transition_observations=True,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "actions": (action_dim,)})
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        updates_per_iteration = args.update_epochs * args.num_minibatches
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
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                network_lr.fill_((1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate)
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(action.reshape((args.num_envs,) + space.shape))
                with timer.span("normalize_transfer", use_cuda=False):
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, np.asarray(raw_reward) * args.reward_scale, terms, truncs,
                                  transition_observations=transition_obs,
                                  observations=obs_step, actions=action)
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

            with timer.span("targets"), torch.no_grad():
                batch = transfer.upload()
                b_obs.copy_(batch.fields["observations"].flatten(0, 1))
                b_actions.copy_(batch.fields["actions"].flatten(0, 1))
                old_mean, old_std, next_logprob, next_q, next_features, data_q = rollout_statistics(
                    b_obs, b_actions, batch.transition_observations.flatten(0, 1))
                continuing = 1.0 - batch.terminations
                shape = (args.num_steps, args.num_envs)
                # Soft reward: the next state's entropy bonus, absent when there is no next state.
                temperature = actor.log_temperature.exp()
                soft_rewards = batch.rewards - args.gamma * temperature * next_logprob.view(shape) * continuing
                targets = soft_lambda_returns(gae_fn, soft_rewards, next_q.view(shape), batch.terminations,
                                              batch.truncations, args.gamma, args.gve_lambda).flatten().clone()
                b_targets.copy_(targets)
                b_labels.copy_(hl_project(targets))
                b_next_features.copy_(next_features)
                b_continuing.copy_(continuing.flatten())
                b_old_mean.copy_(old_mean)
                b_old_std.copy_(old_std)
                support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()

            with timer.span("update"):
                if update_steps is None:
                    if args.compile:
                        minibatch.copy_(torch.arange(M, device=device))
                        graphs = capture_steps([critic_step, actor_step],
                                               optimizers=[critic_optimizer, actor_optimizer],
                                               state_tensors=[critic_sums, actor_sums, grad_sums])
                        update_steps = [graph.replay for graph in graphs]
                    else:
                        update_steps = [critic_step, actor_step]
                critic_sums.zero_()
                actor_sums.zero_()
                grad_sums.zero_()
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(B, M, device, shuffle_generator):
                        minibatch.copy_(indices)
                        for update_step in update_steps:
                            update_step()

            scale = 1.0 / updates_per_iteration
            logged = gather_metrics({
                **{f"losses/{name}": critic_sums[i] * scale for i, name in enumerate(CRITIC_METRICS)},
                **{f"policy/{name}": actor_sums[i] * scale for i, name in enumerate(ACTOR_METRICS)},
                "grad/actor_preclip_norm": grad_sums[0] * scale,
                "grad/critic_preclip_norm": grad_sums[1] * scale,
                "losses/explained_variance": explained_variance(data_q, targets),
                "values/target_mean": targets.mean(),
                "values/target_max": targets.max(),
                "values/support_clip_frac": support_clip,
                "values/next_logprob": next_logprob.mean(),
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite REPPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
                              if args.anneal_lr else args.learning_rate, global_step)
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
