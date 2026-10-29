# REPPO v10: Retrace-corrected critic replay. v6 showed that replaying stored rollouts with
# uncorrected soft lambda-chains hurts (8.5k fresh-only vs 4.7k with 8 rollouts): stale behavior
# actions enter every chain at full weight lambda, so old rollouts pull Q toward past policies.
# v7 also showed critic data volume matters (halving it hurt). Retrace (Munos et al. 2016) keeps the
# data and removes the bias:
#   G_t = r_t - gamma T log pi(a'|s') + gamma [ Q'(s', a' ~ pi) + c_{t+1} (G_{t+1} - Q'(s', a_{t+1})) ]
#   c_{t+1} = lambda min(1, pi(a_{t+1}|s_{t+1}) / mu(a_{t+1}|s_{t+1}))
# with mu the stored behavior density (exact, from the host sampler) and pi, Q' the current actor and
# frozen critic. Ratios are taken in pre-tanh space, where the squash Jacobian cancels. The per-step
# trace is a first-order linear recurrence (cleanrl/shared/linear_scan.py). Differences from v7 at
# --buffer-iterations 1: the correction subtracts Q'(s', a_{t+1}) of the taken action instead of the
# sampled Q'(s', a'), a lower-variance control variate that is also the off-policy-correct one.
# Hypothesis: with correction, replayed rollouts add critic data without bias, so a multi-rollout
# buffer improves Q (and therefore the pathwise actor gradient) instead of degrading it.
import copy
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
from cleanrl.shared.linear_scan import reverse_linear_scan
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics
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
CRITIC_METRICS = ("critic_ce", "aux_loss", "q_mse", "support_clip_frac", "target_mean")
TRACE_METRICS = ("mean_ratio", "fresh_ratio", "oldest_ratio", "cut_frac", "fresh_abs_log_ratio")
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
    gve_lambda: float = 0.95
    """Retrace lambda: the trace is lambda min(1, pi / mu)"""
    critic_updates: int = 32
    """critic updates per rollout, each on a uniform replay minibatch"""
    actor_updates: int = 32
    """actor updates per rollout, interleaved evenly with the critic's"""
    buffer_iterations: int = 4
    """replay capacity in rollouts; 1 = the fresh rollout only"""
    actor_replay: bool = False
    """sample actor states from the whole replay instead of the fresh rollout"""
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
    """lowest supported scaled soft return"""
    v_max: float = 250.0
    """highest supported scaled soft return"""
    aux_loss_coef: float = 1.0
    """weight of the next-state feature prediction loss"""

    # Actor
    actor_hidden_dim: int = 512
    """actor width"""
    actor_min_std: float = 0.0
    """additive floor on the pre-squash standard deviation"""
    kl_bound: float = 0.03
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
    ent_start: float = 0.01
    """initial entropy temperature"""
    ent_target_mult: float = 0.5
    """target entropy is -ent_target_mult * action_dim"""
    dual_step_per_iteration: float = 0.15
    """log-space step budget of each dual variable per iteration (REPPO: 512 updates x 3e-4)"""
    obs_clip: float = 10.0
    """observation normalization clip (VectorObsNorm's)"""

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
    buffer_size: int = 0
    """replay capacity in transitions (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def normed_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), nn.RMSNorm(out_dim, eps=1e-5), nn.SiLU()]


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
        width = args.actor_hidden_dim
        # Flat stack so the fused host mirror can evaluate it in one native call.
        self.net = nn.Sequential(*normed_layer(observation_dim, width), *normed_layer(width, width),
                                 nn.Linear(width, 2 * action_dim))
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(args.ent_start))))
        self.log_kl_multiplier = nn.Parameter(torch.tensor(float(np.log(args.kl_start))))
        self.min_std = args.actor_min_std

    def forward(self, observations):
        return policy_moments(self.net, observations, self.min_std)


def policy_moments(net, observations, min_std):
    mean, log_std = net(observations).chunk(2, dim=-1)
    return mean, log_std.exp() + min_std


class Critic(nn.Module):
    """REPPO's Q(s, a): encoder -> {categorical value head, next-feature predictor}."""

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        width = args.critic_hidden_dim
        self.encoder = nn.Sequential(*normed_layer(observation_dim + action_dim, width), nn.Linear(width, width))
        self.value_head = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, args.num_bins))
        self.predictor = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, width))
        self.hl = HLGaussConfig(v_min=args.v_min, v_max=args.v_max, num_bins=args.num_bins).build(device)
        self.zero_dist = nn.Parameter(self.hl.project(torch.zeros((), device=device)))

    def features(self, observations, actions):
        return self.encoder(torch.cat((observations, actions), dim=-1))

    def logits(self, features):
        return self.value_head(features) + ZERO_DIST_SCALE * self.zero_dist

    def value(self, observations, actions):
        return self.hl.to_scalar(self.logits(self.features(observations, actions)))


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs.

    Besides the action, keeps the pre-tanh sample and its Gaussian log-density: the behavior
    density Retrace's importance ratios are taken against (the tanh Jacobian cancels in the ratio).
    """

    def __init__(self, num_envs, action_dim, min_std):
        shape = (num_envs, action_dim)
        self.min_std = np.float32(min_std)
        self.std = np.empty(shape, dtype=np.float32)
        self.noise = np.empty(shape, dtype=np.float32)
        self.pre = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)
        self.logprob = np.empty(num_envs, dtype=np.float32)
        self.offset = np.float32(action_dim * LOG_SQRT_2PI)

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
        np.exp(log_std, out=self.std)
        self.std += self.min_std
        rng.standard_normal(self.noise.shape, dtype=np.float32, out=self.noise)
        np.multiply(self.noise, self.std, out=self.pre)
        self.pre += mean
        np.tanh(self.pre, out=self.action)
        np.einsum("ij,ij->i", self.noise, self.noise, out=self.logprob)
        self.logprob *= -0.5
        self.logprob -= np.log(self.std).sum(-1)
        self.logprob -= self.offset
        return self.action


def gaussian_logprob(pre, mean, std):
    return (-0.5 * ((pre - mean) / std).square() - std.log() - LOG_SQRT_2PI).sum(-1)


def retrace_returns(soft_rewards, next_q, taken_q, traces, continuing, chain_ends, gamma):
    """Soft Retrace targets over time-major (T, N) tensors.

    G_t = soft_r_t + gamma cont_t [Q'_t + c_{t+1} (G_{t+1} - Qa_{t+1})], where Q'_t = Q(s_{t+1}, a' ~ pi),
    Qa_t = Q(s_t, a_t) of the stored action and c_t its truncated trace. The chain is cut after
    terminations, truncations and rollout ends (``chain_ends``), where G_t bootstraps from Q'_t alone.
    As y_t = x_t + a_t y_{t+1}: a_t = gamma cont_t c_{t+1} (1 - end_t), x_t = soft_r_t + gamma cont_t Q'_t
    - a_t Qa_{t+1}.
    """
    shift = lambda tensor: torch.cat((tensor[1:], torch.zeros_like(tensor[:1])))
    coefficients = gamma * continuing * (1.0 - chain_ends) * shift(traces)
    inputs = soft_rewards + gamma * continuing * next_q - coefficients * shift(taken_q)
    return reverse_linear_scan(inputs, coefficients)


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates,
           args.buffer_iterations) <= 0:
        raise ValueError("environment, rollout, minibatch, update and buffer counts must be positive")
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if not 0.0 <= args.gve_lambda <= 1.0:
        raise ValueError("gve_lambda must lie in [0, 1]")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.buffer_size = args.buffer_iterations * args.batch_size
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
        # Frozen per-iteration copies: the critic's bootstrap target and the KL anchor.
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        old_actor = copy.deepcopy(actor.net).requires_grad_(False)
        actor_network, critic_parameters = list(actor.net.parameters()), list(critic.parameters())
        dual_parameters = [actor.log_temperature, actor.log_kl_multiplier]
        actor_optimizer = torch.optim.AdamW(
            [{"params": actor_network},
             {"params": dual_parameters, "lr": args.dual_step_per_iteration / args.actor_updates,
              "weight_decay": 0.0}],
            lr=args.learning_rate, fused=True)
        critic_optimizer = torch.optim.AdamW(critic_parameters, lr=args.learning_rate, fused=True)
        actor_parameters = actor_network + dual_parameters
        target_entropy = -args.ent_target_mult * action_dim
        B, M, C, N = args.batch_size, args.minibatch_size, args.buffer_size, args.num_envs

        # GPU replay of raw observations. Rows keep the rollout's (step, env) flattening and slots
        # are whole rollouts, so a row's environment is row % num_envs.
        f32 = dict(device=device, dtype=torch.float32)
        replay_obs = torch.zeros((C, observation_dim), **f32)
        replay_next_obs = torch.zeros((C, observation_dim), **f32)
        replay_actions = torch.zeros((C, action_dim), **f32)
        replay_pre_actions = torch.zeros((C, action_dim), **f32)
        replay_behavior_logprob = torch.zeros(C, **f32)
        replay_rewards = torch.zeros(C, **f32)
        replay_continuing = torch.zeros(C, **f32)
        replay_truncated = torch.zeros(C, **f32)
        replay_targets = torch.zeros(C, **f32)
        replay_next_q = torch.zeros(C, **f32)
        replay_next_logprob = torch.zeros(C, **f32)
        replay_taken_q = torch.zeros(C, **f32)
        replay_log_ratio = torch.zeros(C, **f32)
        # Time-major (buffer_iterations * num_steps, num_envs) view of the ring; every stored rollout
        # ends in a bootstrap, so rollouts (and the ring's wrap point) never share a trace.
        time_shape = (C // N, N)
        rollout_ends = torch.zeros(time_shape, **f32)
        rollout_ends[args.num_steps - 1 :: args.num_steps] = 1.0
        rollout_rows = torch.arange(B, device=device)
        obs_mean = torch.zeros((N, observation_dim), **f32)
        obs_inv_std = torch.ones((N, observation_dim), **f32)
        filled = torch.zeros((), **f32)
        fresh_start = torch.zeros((), dtype=torch.long, device=device)
        critic_sums = torch.zeros(len(CRITIC_METRICS), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(2, **f32)

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def observations(rows):
            return normalize(replay_obs[rows], rows)

        def next_observations(rows):
            return normalize(replay_next_obs[rows], rows)

        def trace_inputs(rows):
            """Under the current actor and frozen critic: Q'(s', a' ~ pi), log pi(a'|s'), Q'(s, a) of
            the stored action, and log pi(u|s) - log mu(u|s) of its pre-tanh sample u."""
            obs, next_obs = observations(rows), next_observations(rows)
            next_actions, next_logprob = squashed_sample(*actor(next_obs))
            log_ratio = gaussian_logprob(replay_pre_actions[rows], *actor(obs)) - replay_behavior_logprob[rows]
            return (target_critic.value(next_obs, next_actions), next_logprob,
                    target_critic.value(obs, replay_actions[rows]), log_ratio)

        def critic_loss(rows):
            obs, next_obs = observations(rows), next_observations(rows)
            actions, continuing, targets = replay_actions[rows], replay_continuing[rows], replay_targets[rows]
            with torch.no_grad():
                next_actions, _ = squashed_sample(*actor(next_obs))
                next_features = target_critic.features(next_obs, next_actions)
                labels = critic.hl.project(targets)
            features = critic.features(obs, actions)
            logits = critic.logits(features)
            ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            prediction_error = (critic.predictor(features) - next_features).square().mean(-1)
            aux = (continuing * prediction_error).mean()
            q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            return ce + args.aux_loss_coef * aux, torch.stack((ce.detach(), aux.detach(), q_mse,
                                                              support_clip, targets.mean()))

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs, args.actor_min_std)
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
            trace_inputs = graph_compile(trace_inputs)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        def uniform_rows(count):
            """Uniform replay rows from the default CUDA generator (graph-safe, fresh per replay)."""
            rows = (torch.rand(M, device=device) * count).long()
            return torch.minimum(rows, count.long() - 1)

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            loss, metrics = critic_loss(uniform_rows(filled))
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            critic_sums.add_(metrics)

        fresh_count = torch.full((), float(B), **f32)

        def actor_step():
            if args.actor_replay:
                rows = uniform_rows(filled)
            else:
                rows = fresh_start + uniform_rows(fresh_count)
            actor_optimizer.zero_grad(set_to_none=False)
            loss, metrics = actor_loss(rows)
            # The critic is only differentiated through its action input.
            loss.backward(inputs=actor_parameters)
            grad_sums[0] += nn.utils.clip_grad_norm_(actor_network, args.max_grad_norm, foreach=True)
            actor_optimizer.step()
            actor_sums.add_(metrics)

        update_steps = None  # captured lazily, once the first rollout populated the replay
        critic_per_actor = args.critic_updates // args.actor_updates

        host_actor = make_host_mirror(actor.net, args.num_envs)
        writer.add_text("host_actor", f"fused={host_actor.fused} fallback={host_actor.fallback_reason}")
        sampler = np.random.default_rng(args.seed)
        sample_actions = HostSquashedGaussian(args.num_envs, action_dim, args.actor_min_std)

        def act(observations):
            action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return action

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"raw_obs": obs_shape, "raw_next_obs": obs_shape,
                                           "actions": (action_dim,), "pre_actions": (action_dim,),
                                           "behavior_logprob": ()})
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape, clip=args.obs_clip)
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
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(action.reshape((args.num_envs,) + space.shape))
                with timer.span("normalize_transfer", use_cuda=False):
                    next_obs_np, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    np.copyto(raw_next, raw_obs)
                    for index in np.flatnonzero(np.logical_or(terms, truncs)):
                        raw_next[index] = infos["final_observation"][index]
                    transfer.push(step, np.asarray(raw_reward) * args.reward_scale, terms, truncs,
                                  raw_obs=raw_current, raw_next_obs=raw_next, actions=action,
                                  pre_actions=sample_actions.pre, behavior_logprob=sample_actions.logprob)
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
                slot = ((iteration - 1) % args.buffer_iterations) * B
                rows = slice(slot, slot + B)
                replay_obs[rows].copy_(batch.fields["raw_obs"].flatten(0, 1))
                replay_next_obs[rows].copy_(batch.fields["raw_next_obs"].flatten(0, 1))
                replay_actions[rows].copy_(batch.fields["actions"].flatten(0, 1))
                replay_pre_actions[rows].copy_(batch.fields["pre_actions"].flatten(0, 1))
                replay_behavior_logprob[rows].copy_(batch.fields["behavior_logprob"].flatten())
                replay_rewards[rows].copy_(batch.rewards.flatten())
                replay_continuing[rows].copy_(1.0 - batch.terminations.flatten())
                replay_truncated[rows].copy_(batch.truncations.flatten())
                filled.fill_(float(min(iteration, args.buffer_iterations) * B))
                fresh_start.fill_(slot)
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

            with timer.span("targets"), torch.no_grad():
                # Re-evaluate every stored rollout's Retrace targets under the current policy and critic.
                stored_count = min(iteration, args.buffer_iterations)
                for stored in range(stored_count):
                    rows = slice(stored * B, (stored + 1) * B)
                    next_q, next_logprob, taken_q, log_ratio = trace_inputs(rollout_rows + stored * B)
                    replay_next_q[rows].copy_(next_q)
                    replay_next_logprob[rows].copy_(next_logprob)
                    replay_taken_q[rows].copy_(taken_q)
                    replay_log_ratio[rows].copy_(log_ratio)
                clipped_ratio = replay_log_ratio.clamp(max=0.0).exp()
                traces = args.gve_lambda * clipped_ratio
                temperature = actor.log_temperature.exp()
                soft_rewards = replay_rewards - args.gamma * temperature * replay_next_logprob * replay_continuing
                targets = retrace_returns(
                    soft_rewards.view(time_shape), replay_next_q.view(time_shape), replay_taken_q.view(time_shape),
                    traces.view(time_shape), replay_continuing.view(time_shape),
                    torch.maximum(replay_truncated.view(time_shape), rollout_ends), args.gamma)
                replay_targets.copy_(targets.flatten())
                # The oldest stored rollout is the slot the next insert overwrites (slot 0 until full).
                oldest = (iteration % args.buffer_iterations) * B if iteration >= args.buffer_iterations else 0
                live = slice(0, stored_count * B)
                trace_sums = torch.stack((
                    clipped_ratio[live].mean(), clipped_ratio[slot : slot + B].mean(),
                    clipped_ratio[oldest : oldest + B].mean(), (replay_log_ratio[live] < 0.0).float().mean(),
                    replay_log_ratio[slot : slot + B].abs().mean()))

            with timer.span("update"):
                if update_steps is None:
                    if args.compile:
                        graphs = capture_steps([critic_step, actor_step],
                                               optimizers=[critic_optimizer, actor_optimizer],
                                               state_tensors=[critic_sums, actor_sums, grad_sums])
                        update_steps = [graph.replay for graph in graphs]
                    else:
                        update_steps = [critic_step, actor_step]
                critic_update, actor_update = update_steps
                critic_sums.zero_()
                actor_sums.zero_()
                grad_sums.zero_()
                for update in range(1, args.critic_updates + 1):
                    critic_update()
                    if update % critic_per_actor == 0:
                        actor_update()

            logged = gather_metrics({
                **{f"losses/{name}": critic_sums[i] / args.critic_updates for i, name in enumerate(CRITIC_METRICS)},
                **{f"policy/{name}": actor_sums[i] / args.actor_updates for i, name in enumerate(ACTOR_METRICS)},
                "grad/actor_preclip_norm": grad_sums[0] / args.actor_updates,
                "grad/critic_preclip_norm": grad_sums[1] / args.critic_updates,
                **{f"retrace/{name}": trace_sums[i] for i, name in enumerate(TRACE_METRICS)},
            })
            if any(not np.isfinite(value) for value in logged.values()):
                raise FloatingPointError("nonfinite REPPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
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
