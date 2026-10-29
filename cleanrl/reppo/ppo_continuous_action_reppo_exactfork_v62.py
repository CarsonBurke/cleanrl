# REPPO v62 exact-target fork: v61's Q(lambda)-corrected critic targets, measured from a common training state.
# v61 (from scratch, seed 1): the uncorrected control and lambda 0.8 both fell into the flipped-on-its-back
# trap (~2.1k) that has taken 5 of the last 6 novel runs, while v60-base (same config, differing only in RNG
# consumption) did not, so from-scratch single-seed runs cannot attribute the target. Resuming every arm from
# one upright full-state checkpoint of v60-base (--resume; actor, critic, duals, Adam states, obs statistics)
# pairs the arms past the flip window. Target, see v61: G_t = r_t + gamma [V'_{t+1} + lam (G_{t+1} - Q'_{t+1})]
# has pure critic error under deterministic dynamics, so its fixed point is soft Q^pi for every lambda. The
# diagnostics now sample with their own generator, so --no-exact-targets consumes the training RNG exactly as
# v55/v60 do (its targets differ from v60 only by the Rao-Blackwellized entropy term inside V').
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
DIAGNOSTIC_METRICS = ("dqda_norm", "advantage_corr_heldout", "advantage_std_ratio", "bellman_rms_heldout",
                      "bellman_rms_ratio")
PROBE_METRICS = ("true_curvature", "true_curvature_se", "critic_curvature", "maxent_curvature",
                 "true_curvature_min_dim", "true_curvature_max_dim", "critic_curvature_min_dim",
                 "slope_cross", "slope_cross_se", "slope_energy", "slope_self_cross", "slope_beta",
                 "real_advantage_std", "critic_contrast_std")
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
    exact_targets: bool = True
    """correct each continuation by Q'(s, a) at the taken action (no future-action noise); False: REPPO's target"""
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
    snapshot_steps: str = ""
    """comma-separated global steps (e.g. 4e6,12e6) after which actor, critic and normalization are saved"""
    checkpoint_steps: str = ""
    """comma-separated global steps at which to save a full training-state checkpoint"""
    resume: str = ""
    """full-state checkpoint to resume from; the run then continues for --iterations iterations"""
    iterations: int = 0
    """if positive, the number of iterations to run (overrides --total-timesteps)"""

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

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
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


def exact_lambda_returns(gae_fn, rewards, taken_values, next_values, terminations, truncations, gamma, lam):
    """Q(lambda) target G_t = r_t + gamma [V'_{t+1} + lam (G_{t+1} - Q'(s_{t+1}, a_{t+1}))].

    Explicit-next-value GAE with v_t = Q'(s_t, a_t) at the taken action: A_t = r_t + gamma V'_{t+1} - Q'_t
    + gamma lam A_{t+1} and G_t = A_t + Q'_t, which unrolls to the recursion above. Terminations give r_t,
    truncations (and rollout ends) r_t + gamma V'_{t+1}.
    """
    _, returns = gae_fn(rewards, taken_values, terminations, truncations, next_values, gamma, lam)
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
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    if not 0 < args.diagnostic_rows <= args.batch_size or args.diagnostic_samples < 1 or args.eval_interval < 0:
        raise ValueError("diagnostic rows must fit the rollout; samples positive; eval interval nonnegative")
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
        # Frozen per-iteration copies: the bootstrap target and the KL anchor.
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        old_actor = copy.deepcopy(actor.net).requires_grad_(False)
        # The policy one iteration older than old_actor: the one that chased the critic before the current one.
        previous_actor = copy.deepcopy(actor.net).requires_grad_(False)
        offset_actor = previous_actor if args.offset_reference == "belief" else actor.net
        actor_network, critic_parameters = list(actor.net.parameters()), list(critic.parameters())
        dual_parameters = [actor.log_kl_multiplier] + ([actor.log_dual_temperature] if actor.dual else [])
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

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def observations(rows):
            return normalize(rollout_obs[rows], rows)

        def next_observations(rows):
            return normalize(rollout_next_obs[rows], rows)

        def bootstrap_values(rows, noise=None):
            """Expected soft value V'(s') = mean_k [Q_frozen(s', a'_k) - T log pi(a'_k|s')] over K current-policy
            actions, and the entropy bonus T H^(s') = -T mean_k log pi(a'_k|s')."""
            next_obs = next_observations(rows)
            mean, std = actor(next_obs)
            samples = args.bootstrap_samples
            if noise is None:  # training draws in-graph; diagnostics pass their own generator's draws
                noise = torch.randn((samples,) + mean.shape, device=mean.device)
            pre = mean + std * noise
            next_actions = torch.tanh(pre)
            # Rao-Blackwellized log-density: the Gaussian term's expectation is exact (E[-noise^2 / 2] = -1/2),
            # so only the low-variance tanh Jacobian is sampled. v61 weights V' by 1 instead of 1 - lambda,
            # so this sampling noise enters its targets 1 / (1 - lambda)^2 times more strongly than REPPO's.
            next_logprob = -0.5 - std.log() - LOG_SQRT_2PI - tanh_log_det(pre)
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

        def action_gradient(model, obs, actions):
            actions = actions.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(model.value(obs, actions).sum(), actions)
            return grad

        def critic_diagnostics(rows):
            """Is the critic's action dependence signal? Measured on fresh (held-out) rollout rows.

            advantage_corr  correlation of Q(s, a) - E_pi Q(s, .) with G_t - E_pi Q(s, .) on actions the critic
                            has not been trained on.
            bellman_rms     rms of r + gamma V'(s') - Q(s, a): with deterministic transitions, exactly the critic's
                            one-step inconsistency (up to V's sampled expectation); _ratio divides by the spread
                            of the critic's own advantages, the action signal it must resolve.
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
            next_value, _ = bootstrap_values(rows, torch.randn((args.bootstrap_samples, rows.numel(), action_dim),
                                                               device=device, generator=diagnostic_generator))
            residual = (rollout_rewards[rows] + args.gamma * rollout_continuing[rows] * next_value
                        - target_critic.value(obs, rollout_actions[rows]))
            bellman_rms = residual.square().mean().sqrt()
            return torch.stack((slope.norm(dim=-1).mean(), correlation,
                                centered_c.std() / (centered_r.std() + 1e-12), bellman_rms,
                                bellman_rms / (centered_c.std() + 1e-12)))

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

        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16)

        def encode_targets():
            """f'(s, a) for all rollout rows under the (just refreshed) target critic."""
            target_features.copy_(target_critic.features(observations(rollout_rows), rollout_actions))

        if args.compile:
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

        update_steps = None  # captured lazily, once the first rollout populated the buffers
        # One Triton kernel; a cudagraph-tree wrapper would fault the manually captured update graphs.
        gae_fn = get_gae_fn(compiled=args.compile, mode="default", explicit_next_values=True)
        critic_per_actor = args.critic_updates // args.actor_updates

        host_actor = make_host_mirror(actor.net, args.num_envs)
        writer.add_text("host_actor", f"fused={host_actor.fused} fallback={host_actor.fallback_reason}")
        sampler = np.random.default_rng(args.seed)
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

        def evaluate(episode):
            """One mean-action episode, observations normalized with environment 0's statistics."""
            eval_actor.refresh()
            mean = obs_norm.means[0]
            inv_std = 1.0 / np.sqrt(obs_norm.variances[0] + obs_norm.epsilon)
            raw, _ = eval_env.reset(seed=args.seed + 10_000 + episode)
            total, saturated, steps, done = 0.0, 0.0, 0, False
            while not done:
                normalized = np.clip((raw - mean) * inv_std, -args.obs_clip, args.obs_clip).astype(np.float32)
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
                                           "noise": (action_dim,)})
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape, clip=args.obs_clip)
        step_offset = iteration_offset = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            actor.load_state_dict(checkpoint["actor_state"])
            critic.load_state_dict(checkpoint["critic"])
            target_critic.load_state_dict(checkpoint["critic"])
            old_actor.load_state_dict(checkpoint["old_actor"])
            previous_actor.load_state_dict(checkpoint["previous_actor"])
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            # In place: the native normalizer holds pointers to these arrays.
            np.copyto(obs_norm.means, checkpoint["obs_means"])
            np.copyto(obs_norm.variances, checkpoint["obs_variances"])
            np.copyto(obs_norm.counts, checkpoint["obs_counts"])
            step_offset, iteration_offset = int(checkpoint["global_step"]), int(checkpoint["iteration"])
            host_actor.refresh()
            writer.add_text("resume", f"resume={args.resume} step={step_offset} iteration={iteration_offset}")

        def save_checkpoint(path, global_step, iteration):
            """Full training state, plus the snapshot keys scripts/landscape.py and fork_eval.py read."""
            torch.save({"actor_state": actor.state_dict(), "critic": critic.state_dict(),
                        "old_actor": old_actor.state_dict(), "previous_actor": previous_actor.state_dict(),
                        "actor_optimizer": actor_optimizer.state_dict(),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "actor_net": actor.net.state_dict(), "obs_means": obs_norm.means.copy(),
                        "obs_variances": obs_norm.variances.copy(), "obs_counts": obs_norm.counts.copy(),
                        "obs_epsilon": obs_norm.epsilon, "global_step": global_step, "iteration": iteration,
                        "temperature": actor.temperature().tolist(), "target_entropy": float(target_entropy),
                        "args": vars(args)}, path)

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
                                  noise=sample_actions.noise)
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
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(previous_actor.parameters()), list(old_actor.parameters()))
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

            cuts = torch.maximum(rollout_truncated.view(time_shape), rollout_ends)
            with torch.no_grad():
                # The episode survives step t -> t + 1 inside the rollout unless it terminated, truncated, or
                # the rollout ended at t.
                survives = (rollout_continuing.view(time_shape) * (1.0 - cuts))
                chain = torch.ones_like(survives)
                for k in range(H):
                    chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
                    rollout_chain[k].copy_(chain.flatten())

            def refresh_targets(lam, out, soft=True, exact=False):
                """Soft lambda-returns G_t = r_t + gamma [(1 - lam) V'_{t+1} + lam (G_{t+1} + T H^_{t+1})].

                The lambda-branch's entropy bonus at s_{t+1} is the K-sample expected one (unbiased for the
                taken action's -T log pi, lower variance). It is folded into the reward where the chain
                continues; cut chains (truncation, rollout end) bootstrap from V' alone; terminations
                bootstrap nothing.
                """
                next_values, bonus = bootstrap_values(rollout_rows)
                continuing = rollout_continuing.view(time_shape)
                rewards = rollout_rewards.view(time_shape)
                if exact:
                    taken = target_critic.value(observations(rollout_rows), rollout_actions).view(time_shape)
                    out.copy_(exact_lambda_returns(gae_fn, rewards, taken, next_values.view(time_shape),
                                                   1.0 - continuing, cuts, args.gamma, lam).flatten())
                    return
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

            profiler = None
            if iteration == args.profile_iteration:
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                profiler.__enter__()
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
                    if (update - 1) % args.target_refresh == 0:
                        with torch.no_grad():
                            if update > 1:
                                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                            refresh_targets(args.gve_lambda, rollout_targets, exact=args.exact_targets)
                            encode_targets()
                    critic_update()
                    if update % critic_per_actor == 0:
                        actor_update()

            if profiler is not None:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                with open(f"runs/{run_name}/update_profile.txt", "w") as handle:
                    handle.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40))
            logged = gather_metrics({
                **{f"losses/{name}": critic_sums[i] / args.critic_updates for i, name in enumerate(CRITIC_METRICS)},
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
            if args.eval_interval and (iteration % args.eval_interval == 0 or iteration == last_iteration):
                with timer.span("eval", use_cuda=False):
                    eval_return, eval_saturation = evaluate(iteration)
                writer.add_scalar("eval/deterministic_return", eval_return, global_step)
                writer.add_scalar("eval/deterministic_saturation", eval_saturation, global_step)
            while pending_snapshots and global_step >= pending_snapshots[0]:
                pending_snapshots.pop(0)
                # Everything the oracle needs to rebuild the acting policy and the critic that judges it.
                torch.save({"actor_net": actor.net.state_dict(), "critic": critic.state_dict(),
                            "obs_means": obs_norm.means.copy(), "obs_variances": obs_norm.variances.copy(),
                            "obs_epsilon": obs_norm.epsilon, "global_step": global_step, "iteration": iteration,
                            "temperature": actor.temperature().tolist(), "args": vars(args)},
                           f"runs/{run_name}/snapshot_{global_step}.pt")
            while pending_checkpoints and global_step >= pending_checkpoints[0]:
                pending_checkpoints.pop(0)
                save_checkpoint(f"runs/{run_name}/checkpoint_{global_step}.pt", global_step, iteration)
            if iteration == last_iteration:
                save_checkpoint(f"runs/{run_name}/final.pt", global_step, iteration)
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
