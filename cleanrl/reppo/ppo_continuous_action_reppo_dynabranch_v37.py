# REPPO v37: v36 (v27 H8 on the fast path) whose critic also fits model-imagined, CRN-paired action branches.
# The v29 simulator oracle: the critic's within-state action ranking correlates only 0.15-0.20 with the
# truth. Every real state carries one action, so dQ/da must come from cross-state generalization on
# 16k fresh rows per iteration. Physics one step out is nearly deterministic and policy independent, so:
# - A world-model ensemble M_e(s, a) -> (normalized delta, scaled reward, continuation logit) is fit on
#   a replay of all real transitions (a dynamics model is valid off-policy; the critic never sees replay).
# - At every target refresh, start states are drawn from real data (half this rollout, half replay); each
#   gets A first actions from the current actor and is rolled H_i steps by the model under the current
#   actor. All branches of a start share later policy noise, bootstrap noise and ensemble member (common
#   random numbers), so their soft lambda-return differences isolate the first action's effect.
# - The critic's loss adds HL-Gauss CE to those targets: ~8x the on-policy action-value data per
#   iteration, with A actions per state and contrasts free of return noise. Actor and duals are v27's.
# Hypothesis: better within-state action ranking -> better dQ/da -> faster, longer policy improvement.
import copy
import math
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
CRITIC_METRICS = ("critic_ce", "aux_loss", "reward_loss", "q_mse", "support_clip_frac", "target_mean",
                  "imagined_ce", "imagined_q_mse")
MODEL_METRICS = ("delta_nmse", "reward_nmse", "continue_bce")
IMAGINE_METRICS = ("imagined_target_mean", "branch_target_std", "branch_critic_std", "branch_rank_corr",
                   "model_disagreement")
ACTOR_METRICS = ("actor_objective", "q_pi", "entropy", "kl", "kl_violation", "temperature", "kl_multiplier",
                 "std", "saturation", "mean_abs_action")
DIAGNOSTIC_METRICS = ("dqda_cos_consecutive", "dqda_norm", "advantage_corr_heldout", "advantage_std_ratio",
                      "model_advantage_corr_heldout", "model_delta_r2_heldout", "model_reward_r2_heldout")


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

    # World model and imagined branches
    model_members: int = 5
    """world-model ensemble size"""
    model_hidden_dim: int = 512
    """world-model width"""
    model_rows: int = 1024
    """replay rows per member per model update (one model update per critic update)"""
    replay_capacity: int = 2_000_000
    """real transitions kept for the world model"""
    imagine_starts: int = 4096
    """real start states per target refresh"""
    imagine_fresh_frac: float = 0.5
    """fraction of start states drawn from this rollout (the rest from the whole replay)"""
    imagine_branches: int = 4
    """first actions per start state (A)"""
    imagine_horizon: int = 5
    """imagined steps per branch (H_i); the last bootstraps from the target critic"""
    imagined_rows: int = 2048
    """imagined rows per critic minibatch"""
    imagined_loss_coef: float = 1.0
    """weight of the imagined rows' CE"""

    # Actor
    actor_hidden_dim: int = 512
    """actor width"""
    ent_start: float = 0.01
    """initial entropy temperature"""
    ent_target_mult: float = 0.5
    """target entropy is -ent_target_mult * action_dim"""
    kl_bound: float = 0.03
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
    dual_step_per_iteration: float = 0.15
    """log-space step budget of each dual variable per iteration (REPPO: 512 updates x 3e-4)"""
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

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile the losses and capture both update steps as CUDA graphs"""
    matmul_precision: str = "high"
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


def squashed_sample(mean, std):
    """Reparameterized tanh-Gaussian sample and its exact log-density."""
    return squashed_from_noise(mean, std, torch.randn_like(mean))


def squashed_from_noise(mean, std, noise):
    """tanh(mean + std * noise) and its exact log-density, for caller-supplied (e.g. shared) noise."""
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

    def forward(self, observations):
        return policy_moments(self.net, observations)


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


class EnsembleLinear(nn.Module):
    """E independent affine maps applied to an (E, rows, in) batch."""

    def __init__(self, members, in_dim, out_dim):
        super().__init__()
        bound = 1.0 / math.sqrt(in_dim)  # nn.Linear's default initialization, per member
        self.weight = nn.Parameter(torch.empty(members, in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, out_dim).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleRMSNorm(nn.Module):
    def __init__(self, members, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(members, 1, dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class WorldModel(nn.Module):
    """Ensemble M_e(s, a) -> (normalized observation delta, scaled reward, continuation logit)."""

    def __init__(self, observation_dim, action_dim, args):
        super().__init__()
        members, width = args.model_members, args.model_hidden_dim
        self.hidden = nn.ModuleList([EnsembleLinear(members, observation_dim + action_dim, width),
                                     EnsembleLinear(members, width, width)])
        self.norms = nn.ModuleList([EnsembleRMSNorm(members, width), EnsembleRMSNorm(members, width)])
        self.out = EnsembleLinear(members, width, observation_dim + 2)

    def forward(self, observations, actions):
        """(E, rows, .) inputs -> per-member predictions."""
        x = torch.cat((observations, actions), dim=-1)
        for layer, norm in zip(self.hidden, self.norms):
            x = F.silu(norm(layer(x)))
        out = self.out(x)
        return out[..., :-2], out[..., -2], out[..., -1]


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs."""

    def __init__(self, num_envs, action_dim):
        shape = (num_envs, action_dim)
        self.std = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
        np.exp(log_std, out=self.std)
        rng.standard_normal(self.action.shape, dtype=np.float32, out=self.action)
        self.action *= self.std
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
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates) <= 0:
        raise ValueError("environment, rollout, minibatch and update counts must be positive")
    if args.aux_horizon <= 0 or args.aux_horizon >= args.num_steps:
        raise ValueError("aux_horizon must be in [1, num_steps)")
    if args.target_refresh <= 0 or args.critic_updates % args.target_refresh or args.bootstrap_samples <= 0:
        raise ValueError("target_refresh must divide critic_updates; bootstrap_samples positive")
    if min(args.model_members, args.model_rows, args.imagine_starts, args.imagine_branches,
           args.imagine_horizon, args.imagined_rows) <= 0 or args.imagined_loss_coef < 0.0:
        raise ValueError("model and imagination sizes must be positive")
    if args.imagine_branches < 2:
        raise ValueError("branch contrasts need imagine_branches >= 2")
    if not 0.0 <= args.imagine_fresh_frac <= 1.0:
        raise ValueError("imagine_fresh_frac must be in [0, 1]")
    if args.ent_start <= 0.0:
        raise ValueError("the initial entropy temperature must be positive")
    args.snapshot_at = sorted(int(float(step)) for step in args.snapshot_steps.split(",") if step.strip())
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    if args.replay_capacity < args.batch_size:
        raise ValueError("the replay must hold at least one rollout")
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
        # Frozen per-iteration copies: the bootstrap target, the KL anchor, and the critic one iteration
        # older than the target (the diagnostics compare their action gradients).
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        previous_critic = copy.deepcopy(critic).requires_grad_(False)
        # In-iteration refreshes move target_critic, so the last iteration's final critic is kept separately.
        final_critic = copy.deepcopy(critic).requires_grad_(False)
        old_actor = copy.deepcopy(actor.net).requires_grad_(False)
        actor_network, critic_parameters = list(actor.net.parameters()), list(critic.parameters())
        dual_parameters = [actor.log_temperature, actor.log_kl_multiplier]
        actor_optimizer = torch.optim.AdamW(
            [{"params": actor_network},
             {"params": dual_parameters, "lr": args.dual_step_per_iteration / args.actor_updates,
              "weight_decay": 0.0}],
            lr=args.learning_rate, fused=True)
        critic_optimizer = torch.optim.AdamW(critic_parameters, lr=args.learning_rate, fused=True)
        world_model = WorldModel(observation_dim, action_dim, args).to(device)
        model_parameters = list(world_model.parameters())
        model_optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate, fused=True)
        actor_parameters = actor_network + dual_parameters
        target_entropy = -args.ent_target_mult * action_dim
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
        time_shape = (args.num_steps, N)
        rollout_ends = torch.zeros(time_shape, **f32)
        rollout_ends[-1] = 1.0
        rollout_rows = torch.arange(B, device=device)
        obs_mean = torch.zeros((N, observation_dim), **f32)
        obs_inv_std = torch.ones((N, observation_dim), **f32)
        critic_sums = torch.zeros(len(CRITIC_METRICS), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(3, **f32)
        model_sums = torch.zeros(len(MODEL_METRICS), **f32)
        imagine_sums = torch.zeros(len(IMAGINE_METRICS), **f32)
        rollout_env = rollout_rows % N

        # Replay of every real transition, for the world model only (raw observations; normalized at use
        # with the current statistics of the row's environment).
        C, E = args.replay_capacity, args.model_members
        replay_obs = torch.zeros((C, observation_dim), **f32)
        replay_next_obs = torch.zeros((C, observation_dim), **f32)
        replay_actions = torch.zeros((C, action_dim), **f32)
        replay_rewards = torch.zeros(C, **f32)
        replay_continuing = torch.zeros(C, **f32)
        replay_env = torch.zeros(C, dtype=torch.long, device=device)
        replay_size = torch.zeros((), dtype=torch.long, device=device)
        replay_head = torch.zeros((), dtype=torch.long, device=device)  # next write slot
        replay_fill, replay_write = 0, 0
        # Imagined branch rows (normalized observations, first actions, soft lambda-return targets),
        # regenerated at every target refresh; imagine_weight is 0 until the model has seen one rollout.
        Hi, A, S = args.imagine_horizon, args.imagine_branches, args.imagine_starts
        SA, K = A * S, args.bootstrap_samples
        fresh_starts = int(round(args.imagine_fresh_frac * S))
        imagined_obs = torch.zeros((SA, observation_dim), **f32)
        imagined_actions = torch.zeros((SA, action_dim), **f32)
        imagined_targets = torch.zeros(SA, **f32)
        imagine_weight = torch.zeros((), **f32)
        imagined_last = torch.tensor(SA - 1, device=device)
        diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed)

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def normalize_env(raw, env):
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
            bonus = -actor.log_temperature.exp() * next_logprob.mean(0)
            return q.mean(0) + bonus, bonus

        def critic_loss(rows):
            obs, actions, targets = observations(rows), rollout_actions[rows], rollout_targets[rows]
            # Successor rows t + k (k = 1..H), clamped in range; invalid ones are masked by the chain.
            later = torch.minimum(rows + N * torch.arange(1, H + 1, device=device)[:, None], rollout_rows[-1])
            chain = rollout_chain[:, rows]
            later_actions = rollout_actions[later]
            imagined = torch.minimum((torch.rand(args.imagined_rows, device=device) * SA).long(),
                                     imagined_last)
            imagined_target = imagined_targets[imagined]
            with torch.no_grad():
                later_features = target_features[later]
                labels = critic.hl.project(targets)
                imagined_labels = critic.hl.project(imagined_target)
            # One encoder/value pass over the real and imagined rows; the latent roll uses the real ones.
            all_features = critic.features(torch.cat((obs, imagined_obs[imagined])),
                                           torch.cat((actions, imagined_actions[imagined])))
            all_logits = critic.logits(all_features)
            features, logits, imagined_logits = all_features[:M], all_logits[:M], all_logits[M:]
            ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            imagined_ce = -(imagined_labels * imagined_logits.log_softmax(dim=-1)).sum(-1).mean()
            imagined_q_mse = (critic.hl.to_scalar(imagined_logits.detach()) - imagined_target).square().mean()
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
            loss = (ce + args.aux_loss_coef * aux + args.reward_loss_coef * reward_loss
                    + args.imagined_loss_coef * imagine_weight * imagined_ce)
            return loss, torch.stack((ce.detach(), aux.detach(), reward_loss.detach(), q_mse, support_clip,
                                      targets.mean(), imagined_ce.detach(), imagined_q_mse))

        def model_loss():
            """Per-member variance-normalized regression on independent uniform replay rows (unit-free, so
            the small scaled reward is not drowned by the observation deltas)."""
            rows = torch.minimum((torch.rand((E, args.model_rows), device=device) * replay_size).long(),
                                 replay_size - 1)
            env = replay_env[rows]
            # Clipped-space delta, so that imagination's clamp(s + delta) reproduces the next observation.
            obs = normalize_env(replay_obs[rows], env)
            delta_target = normalize_env(replay_next_obs[rows], env) - obs
            rewards, continuing = replay_rewards[rows], replay_continuing[rows]
            delta, reward, continue_logit = world_model(obs, replay_actions[rows])
            delta_nmse = ((delta.float() - delta_target).square().mean(1)
                          / delta_target.var(dim=1).clamp_min(1e-6)).mean()
            reward_nmse = ((reward.float() - rewards).square().mean(1) / rewards.var(dim=1).clamp_min(1e-6)).mean()
            continue_bce = F.binary_cross_entropy_with_logits(continue_logit.float(), continuing)
            return delta_nmse + reward_nmse + continue_bce, torch.stack((
                delta_nmse.detach(), reward_nmse.detach(), continue_bce.detach()))

        def soft_value(obs, noise):
            """V'(s) = mean_k [Q_target(s, a'_k) - T log pi(a'_k|s)] for given noise (K, ..., action_dim), and
            its entropy bonus."""
            mean, std = actor(obs)
            actions, logprob = squashed_from_noise(mean, std, noise)
            q = target_critic.value(obs.expand(noise.shape[0], *obs.shape), actions)
            bonus = -actor.log_temperature.exp() * logprob.mean(0)
            return q.mean(0) + bonus, bonus

        def branch_returns(obs0, first):
            """Soft lambda-returns of imagined branches (A, S) from starts obs0 (S, D) and first actions
            (A, S, action_dim). Later policy noise, bootstrap noise and the ensemble member of each step are
            shared by a start's branches (common random numbers). Same recursion as refresh_targets:
            G_k = r_k + gamma c_k [(1 - lam) V'_{k+1} + lam (G_{k+1} + T H^_{k+1})], G_{H-1} = r + gamma c V'_H."""
            branches, starts = first.shape[:2]
            state, action = obs0.expand(branches, starts, -1), first
            policy_noise = torch.randn((Hi, starts, action_dim), device=device)
            bootstrap_noise = torch.randn((Hi, K, 1, starts, action_dim), device=device)
            members = torch.randint(E, (Hi, 1, 1, starts, 1), device=device)
            rewards, continues, values, bonuses, disagreement = [], [], [], [], []
            for k in range(Hi):
                delta, reward, logit = world_model(state.reshape(1, -1, observation_dim).expand(E, -1, -1),
                                                   action.reshape(1, -1, action_dim).expand(E, -1, -1))
                delta = delta.view(E, branches, starts, observation_dim)
                disagreement.append(delta.std(0).mean())
                pick = members[k]
                delta = delta.gather(0, pick.expand(1, branches, starts, observation_dim))[0]
                reward = reward.view(E, branches, starts).gather(0, pick[..., 0].expand(1, branches, starts))[0]
                logit = logit.view(E, branches, starts).gather(0, pick[..., 0].expand(1, branches, starts))[0]
                state = (state + delta).clamp(-args.obs_clip, args.obs_clip)  # the critic's training region
                value, bonus = soft_value(state, bootstrap_noise[k])
                rewards.append(reward)
                continues.append(torch.sigmoid(logit))
                values.append(value)
                bonuses.append(bonus)
                if k + 1 < Hi:
                    mean, std = actor(state)
                    action = torch.tanh(mean + std * policy_noise[k])
            returns = rewards[-1] + args.gamma * continues[-1] * values[-1]
            lam = args.gve_lambda
            for k in reversed(range(Hi - 1)):
                returns = rewards[k] + args.gamma * continues[k] * (
                    (1.0 - lam) * values[k] + lam * (returns + bonuses[k]))
            return returns, torch.stack(disagreement).mean()

        def imagine():
            """Fresh imagined branch rows from real start states under the live actor and target critic."""
            fresh = (replay_head - B + torch.minimum((torch.rand(fresh_starts, device=device) * B).long(),
                                                     rollout_rows[-1])) % C
            old = torch.minimum((torch.rand(S - fresh_starts, device=device) * replay_size).long(), replay_size - 1)
            rows = torch.cat((fresh, old))
            obs0 = normalize_env(replay_obs[rows], replay_env[rows])
            mean, std = actor(obs0)
            first, _ = squashed_sample(mean.expand(A, -1, -1), std.expand(A, -1, -1))
            targets, disagreement = branch_returns(obs0, first)
            imagined_obs.copy_(obs0.expand(A, -1, -1).reshape(SA, observation_dim))
            imagined_actions.copy_(first.reshape(SA, action_dim))
            imagined_targets.copy_(targets.reshape(SA))
            # Does the target critic already rank a start's branches as the model does?
            q = target_critic.value(obs0.expand(A, -1, -1), first)
            centered_t, centered_q = targets - targets.mean(0), q - q.mean(0)
            rank_corr = (centered_t * centered_q).sum(0) / (centered_t.norm(dim=0) * centered_q.norm(dim=0) + 1e-8)
            imagine_sums.add_(torch.stack((targets.mean(), targets.std(0).mean(), q.std(0).mean(),
                                           rank_corr.mean(), disagreement)))

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs)
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
                                   1.0 - inside.float().mean(), temperature, kl_multiplier, std.mean(),
                                   (actions.abs() > 0.99).float().mean(), torch.tanh(mean).abs().mean())).detach()
            return policy_loss + temperature_loss + multiplier_loss, metrics

        def action_gradient(model, obs, actions):
            actions = actions.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(model.value(obs, actions).sum(), actions)
            return grad

        def critic_diagnostics(rows):
            """Is the critic's action dependence signal? Measured on fresh (held-out) rollout rows.

            dqda_cos_consecutive  cosine of dQ/da at the policy mean between the critic that is fit to
                                  this rollout's predecessor (target) and the one before it: near 1 means
                                  the actor climbs a stable surface, near 0 that one iteration of fitting
                                  rewrites its direction.
            advantage_corr        correlation of Q(s, a) - E_pi Q(s, .) with G_t - E_pi Q(s, .) on actions
                                  the critic has not been trained on.
            """
            obs = observations(rows)
            mean, std = actor(obs)
            deterministic = torch.tanh(mean)
            with torch.enable_grad():
                current = action_gradient(target_critic, obs, deterministic)
                older = action_gradient(previous_critic, obs, deterministic)
            cosine = nn.functional.cosine_similarity(current, older, dim=-1, eps=1e-12).mean()
            samples = args.diagnostic_samples
            noise = torch.randn((samples,) + mean.shape, device=device, generator=diagnostic_generator)
            sampled = torch.tanh(mean + std * noise)
            baseline = target_critic.value(obs.expand(samples, -1, -1), sampled).mean(0)
            critic_advantage = target_critic.value(obs, rollout_actions[rows]) - baseline
            return_advantage = diagnostic_targets[rows] - baseline
            centered_c = critic_advantage - critic_advantage.mean()
            centered_r = return_advantage - return_advantage.mean()
            correlation = (centered_c * centered_r).mean() / (centered_c.std() * centered_r.std() + 1e-12)
            # The model's ranking of the taken action against policy samples, scored the same way; and the
            # model's one-step accuracy on this rollout, which it has not been trained on yet.
            first = torch.cat((rollout_actions[rows][None], sampled))
            model_returns, _ = branch_returns(obs, first)
            model_baseline = model_returns[1:].mean(0)
            model_c = model_returns[0] - model_baseline
            model_r = diagnostic_targets[rows] - model_baseline
            model_c, model_r = model_c - model_c.mean(), model_r - model_r.mean()
            model_correlation = (model_c * model_r).mean() / (model_c.std() * model_r.std() + 1e-12)
            delta_target = next_observations(rows) - obs
            delta, reward, _ = world_model(obs.expand(E, -1, -1), rollout_actions[rows].expand(E, -1, -1))
            delta_r2 = 1.0 - ((delta.mean(0) - delta_target).square().mean(0) / delta_target.var(0).clamp_min(1e-6)).mean()
            rewards = rollout_rewards[rows]
            reward_r2 = 1.0 - (reward.mean(0) - rewards).square().mean() / rewards.var().clamp_min(1e-6)
            return torch.stack((cosine, current.norm(dim=-1).mean(), correlation,
                                centered_c.std() / (centered_r.std() + 1e-12), model_correlation, delta_r2, reward_r2))

        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16)

        def encode_targets():
            """f'(s, a) for all rollout rows under the (just refreshed) target critic."""
            target_features.copy_(target_critic.features(observations(rollout_rows), rollout_actions))

        if args.compile:
            encode_targets = graph_compile(encode_targets)
            bootstrap_values = graph_compile(bootstrap_values)
            critic_loss = graph_compile(critic_loss)
            model_loss = graph_compile(model_loss)
            imagine = graph_compile(imagine)
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

        def model_step():
            model_optimizer.zero_grad(set_to_none=False)
            with autocast():
                loss, metrics = model_loss()
            loss.backward()
            grad_sums[2] += nn.utils.clip_grad_norm_(model_parameters, args.max_grad_norm, foreach=True)
            model_optimizer.step()
            model_sums.add_(metrics)

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
                                   fields={"raw_obs": obs_shape, "raw_next_obs": obs_shape, "actions": (action_dim,)})
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
        pending_snapshots = [step for step in args.snapshot_at if step > global_step]

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
                                  raw_obs=raw_current, raw_next_obs=raw_next, actions=action)
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
                rollout_rewards.copy_(batch.rewards.flatten())
                rollout_continuing.copy_(1.0 - batch.terminations.flatten())
                rollout_truncated.copy_(batch.truncations.flatten())
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(previous_critic.parameters()), list(final_critic.parameters()))
                torch._foreach_copy_(list(final_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)
                slots = torch.arange(replay_write, replay_write + B, device=device) % C
                replay_obs.index_copy_(0, slots, rollout_obs)
                replay_next_obs.index_copy_(0, slots, rollout_next_obs)
                replay_actions.index_copy_(0, slots, rollout_actions)
                replay_rewards.index_copy_(0, slots, rollout_rewards)
                replay_continuing.index_copy_(0, slots, rollout_continuing)
                replay_env.index_copy_(0, slots, rollout_env)
                replay_write, replay_fill = (replay_write + B) % C, min(replay_fill + B, C)
                replay_head.fill_(replay_write)
                replay_size.fill_(replay_fill)
                # The model has trained on at least one full rollout from the second iteration on.
                imagine_weight.fill_(float(iteration > 1))

            cuts = torch.maximum(rollout_truncated.view(time_shape), rollout_ends)
            with torch.no_grad():
                # The episode survives step t -> t + 1 inside the rollout unless it terminated, truncated, or
                # the rollout ended at t.
                survives = (rollout_continuing.view(time_shape) * (1.0 - cuts))
                chain = torch.ones_like(survives)
                for k in range(H):
                    chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
                    rollout_chain[k].copy_(chain.flatten())

            def refresh_targets(lam, out):
                """Soft lambda-returns G_t = r_t + gamma [(1 - lam) V'_{t+1} + lam (G_{t+1} + T H^_{t+1})].

                The lambda-branch's entropy bonus at s_{t+1} is the K-sample expected one (unbiased for the
                taken action's -T log pi, lower variance). It is folded into the reward where the chain
                continues; cut chains (truncation, rollout end) bootstrap from V' alone; terminations
                bootstrap nothing.
                """
                next_values, bonus = bootstrap_values(rollout_rows)
                continuing = rollout_continuing.view(time_shape)
                rewards = rollout_rewards.view(time_shape) + args.gamma * lam * continuing * (1.0 - cuts) * bonus.view(time_shape)
                out.copy_(lambda_returns(gae_fn, rewards, next_values.view(time_shape), 1.0 - continuing, cuts,
                                         args.gamma, lam).flatten())

            with timer.span("targets"), torch.no_grad():
                refresh_targets(args.diagnostic_lambda, diagnostic_targets)

            with timer.span("diagnostics"), torch.no_grad():
                # The actor still equals the rollout's behavior policy (no update yet this iteration).
                diagnostic_rows = torch.randperm(B, device=device, generator=diagnostic_generator)[: args.diagnostic_rows]
                diagnostics = critic_diagnostics(diagnostic_rows)

            profiler = None
            if iteration == args.profile_iteration:
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                profiler.__enter__()
            with timer.span("update"):
                if update_steps is None:
                    if args.compile:
                        graphs = capture_steps([critic_step, actor_step, model_step],
                                               optimizers=[critic_optimizer, actor_optimizer, model_optimizer],
                                               state_tensors=[critic_sums, actor_sums, grad_sums, model_sums])
                        update_steps = [graph.replay for graph in graphs]
                    else:
                        update_steps = [critic_step, actor_step, model_step]
                critic_update, actor_update, model_update = update_steps
                critic_sums.zero_()
                actor_sums.zero_()
                grad_sums.zero_()
                model_sums.zero_()
                imagine_sums.zero_()
                for update in range(1, args.critic_updates + 1):
                    if (update - 1) % args.target_refresh == 0:
                        with torch.no_grad():
                            if update > 1:
                                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                            refresh_targets(args.gve_lambda, rollout_targets)
                            encode_targets()
                            if iteration > 1:
                                imagine()
                    model_update()
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
                "grad/actor_preclip_norm": grad_sums[0] / args.actor_updates,
                "grad/critic_preclip_norm": grad_sums[1] / args.critic_updates,
                "grad/model_preclip_norm": grad_sums[2] / args.critic_updates,
                **{f"model/{name}": model_sums[i] / args.critic_updates for i, name in enumerate(MODEL_METRICS)},
                **{f"imagine/{name}": imagine_sums[i] / (args.critic_updates // args.target_refresh)
                   for i, name in enumerate(IMAGINE_METRICS)},
            })
            if any(not np.isfinite(value) for value in logged.values()):
                raise FloatingPointError("nonfinite REPPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            if args.eval_interval and (iteration % args.eval_interval == 0 or iteration == args.num_iterations):
                with timer.span("eval", use_cuda=False):
                    eval_return, eval_saturation = evaluate(iteration)
                writer.add_scalar("eval/deterministic_return", eval_return, global_step)
                writer.add_scalar("eval/deterministic_saturation", eval_saturation, global_step)
            while pending_snapshots and global_step >= pending_snapshots[0]:
                pending_snapshots.pop(0)
                # Everything the oracle needs to rebuild the acting policy and the critic that judges it.
                torch.save({"actor_net": actor.net.state_dict(), "critic": critic.state_dict(),
                            "world_model": world_model.state_dict(), "trainer_file": __file__,
                            "obs_means": obs_norm.means.copy(), "obs_variances": obs_norm.variances.copy(),
                            "obs_epsilon": obs_norm.epsilon, "global_step": global_step, "iteration": iteration,
                            "temperature": float(actor.log_temperature.exp()), "args": vars(args)},
                           f"runs/{run_name}/snapshot_{global_step}.pt")
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
