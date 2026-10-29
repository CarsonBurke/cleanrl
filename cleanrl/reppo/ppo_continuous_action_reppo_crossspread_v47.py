# REPPO v47: the Beta policy's spread takes its pathwise gradient from a critic the mode did not climb.
# PPO's Beta policy balances exploration without an entropy bonus: its spread only drifts when real
# outcomes correlate with deviation from the mode. REPPO's pathwise spread gradient
# d/dnu E Q(s, a(m, nu, eps)) instead reads the critic's curvature at the mode, and the mode is driven to
# that same critic's argmax, so the curvature there is negative by construction (the optimizer's curse on
# curvature: spurious critic peaks look sharp) and the spread collapses (v19: Beta, no entropy terms,
# entropy -> -13.7, stalled at 7.0k).
# Fix, double estimation as in Double Q: the mode climbs one critic; the spread climbs an independent
# critic whose noise peaks are not aligned with the mode:
#   loss = -(Q_mode(s, a(m, sg nu)) + Q_spread(s, a(sg m, nu))) under v36's per-state KL(old || new) gate.
# --spread-critic:
#   crossfold    (default) envs split in two folds (first/second half, whole trajectories); Q_k fits only
#                fold k. An actor row of fold k takes its mode gradient from Q_k, its spread gradient from
#                the out-of-fold Q_{1-k}. Each critic makes v36's 128 updates of v36's 2048-row minibatch,
#                drawn from its own fold (so twice v36's reuse of each row).
#   independent  both critics fit every row (independent init and minibatches); mode Q_A, spread Q_B.
#   same         control: one critic for both (v19's policy objective on the v36 critic).
# Each critic has its own latent-roll aux, target network and lambda-returns bootstrapped only from its own
# target: E_{a'~pi} Q_target(s', a'). No entropy bonus, target or temperature anywhere; the KL
# multiplier dual is v36's. Base: v36 (v27 H8 latent-roll critic), strict FP32 by default.
# Hypothesis: with an unbiased curvature signal the spread stops collapsing, and returns pass v19/v36.
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
from torch.distributions import Beta
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
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))
LOG_2 = float(np.log(2.0))
NATIVE_EPSILON = 1e-6  # native Beta samples are clipped to [eps, 1 - eps], as make_beta_sampler does
ZERO_DIST_SCALE = 40.9  # REPPO's fixed multiplier on the learned zero-value logit offset
# (mode critic, spread critic) of actor rows from env fold 0 and from env fold 1.
CRITIC_ROLES = {"crossfold": ((0, 1), (1, 0)), "independent": ((0, 1), (0, 1)), "same": ((0, 0), (0, 0))}
CRITIC_METRICS = ("critic_ce", "aux_loss", "reward_loss", "q_mse", "support_clip_frac", "target_mean")
ACTOR_METRICS = ("actor_objective", "q_mode", "q_spread", "entropy", "nu", "mode_abs", "std", "kl",
                 "kl_violation", "kl_multiplier", "saturation")
DIAGNOSTIC_METRICS = ("dqda_cos_consecutive", "dqda_cos_critics", "dqda_norm", "advantage_corr_heldout",
                      "advantage_std_ratio", "spread_grad_mode_critic", "spread_grad_spread_critic",
                      "spread_grad_narrowing_frac")


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
    """AdamW learning rate of both the actor and the critics"""
    num_envs: int = 16
    """the number of parallel game environments (even: two folds)"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    minibatch_size: int = 2048
    """rows per critic update (each critic) and per actor update (half from each fold)"""
    gve_lambda: float = 0.8
    """lambda of the lambda-return critic target"""
    target_refresh: int = 16
    """critic updates between recomputations of the targets from the live critics"""
    bootstrap_samples: int = 4
    """current-policy actions averaged into each bootstrap E_{a'~pi} Q_target(s', a')"""
    diagnostic_lambda: float = 0.95
    """lambda of the held-out diagnostic returns (fixed across --gve-lambda)"""
    critic_updates: int = 128
    """critic updates per rollout, each critic on its own uniform minibatch of its rows"""
    actor_updates: int = 32
    """actor updates per rollout, interleaved evenly with the critics'"""
    max_grad_norm: float = 0.5
    """per-network gradient clipping norm"""
    reward_scale: float = 0.1
    """fixed reward multiplier (keeps the categorical support stationary)"""

    # Critics
    spread_critic: str = "crossfold"
    """which critic the spread climbs: crossfold (out-of-fold critic), independent (second critic), same"""
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
    kl_bound: float = 0.03
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
    dual_step_per_iteration: float = 0.15
    """log-space step budget of the KL dual per iteration (REPPO: 512 updates x 3e-4)"""
    obs_clip: float = 10.0
    """observation normalization clip (VectorObsNorm's)"""

    # Diagnostics
    diagnostic_rows: int = 4096
    """fresh-rollout rows of the per-iteration critic diagnostics"""
    diagnostic_samples: int = 8
    """policy samples in the diagnostics' state baseline and spread-gradient probe"""
    eval_interval: int = 25
    """iterations between deterministic (mode-action) evaluation episodes; 0 disables"""
    snapshot_steps: str = ""
    """comma-separated global steps (e.g. 4e6,12e6) after which actor, critics and normalization are saved"""

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile the losses and capture both update steps as CUDA graphs"""
    matmul_precision: str = "highest"
    """torch float32 matmul precision: highest (strict FP32; TF32 degrades REPPO), high (TF32)"""
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


def policy_logits(net, observations):
    """(z_m, z_nu) per action dimension, in FP32 also under autocast."""
    return net(observations).float().chunk(2, dim=-1)


def beta_concentrations(z_mode, z_precision):
    """alpha = 1 + m nu, beta = 1 + (1 - m) nu with m = sigmoid(z_m), nu = exp(z_nu).

    sigmoid(-z_m) is 1 - m without float32 cancellation near the bounds (the host sampler's formula).
    """
    precision = z_precision.exp()
    return 1.0 + torch.sigmoid(z_mode) * precision, 1.0 + torch.sigmoid(-z_mode) * precision


def beta_actions(alpha, beta, reparameterized):
    """Physical actions 2 x - 1 (implicit reparameterization when reparameterized)."""
    policy = Beta(alpha, beta, validate_args=False)
    native = policy.rsample() if reparameterized else policy.sample()
    return 2.0 * native.clamp(NATIVE_EPSILON, 1.0 - NATIVE_EPSILON) - 1.0


def physical_entropy(alpha, beta):
    """Entropy of 2 x - 1, x ~ Beta(alpha, beta), summed over action dimensions."""
    return (Beta(alpha, beta, validate_args=False).entropy() + LOG_2).sum(-1)


def beta_kl(old_alpha, old_beta, alpha, beta):
    """Closed-form KL(Beta(old) || Beta(new)) summed over dimensions; the affine action map cancels.

    Evaluated in float64: the lgamma terms are O(c log c) and cancel to a KL of O(kl_bound), which in
    float32 is off by ~5% already at alpha + beta ~ 500 (the trust-region mask and dual read this value).
    """
    old_alpha, old_beta, alpha, beta = old_alpha.double(), old_beta.double(), alpha.double(), beta.double()
    old_total = old_alpha + old_beta
    log_norm = (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
                - torch.lgamma(old_alpha) - torch.lgamma(old_beta) + torch.lgamma(old_total))
    return (log_norm + (old_alpha - alpha) * torch.digamma(old_alpha) + (old_beta - beta) * torch.digamma(old_beta)
            + (alpha + beta - old_total) * torch.digamma(old_total)).sum(-1).float()


def beta_std(alpha, beta):
    """Physical-action standard deviation of 2 x - 1."""
    total = alpha + beta
    return 2.0 * (alpha * beta / (total.square() * (total + 1.0))).sqrt()


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, args):
        super().__init__()
        width = args.actor_hidden_dim
        # Flat stack so the fused host mirror can evaluate it in one native call.
        self.net = nn.Sequential(*normed_layer(observation_dim, width), *normed_layer(width, width),
                                 nn.Linear(width, 2 * action_dim))
        head = self.net[-1]
        with torch.no_grad():
            # Near-state-independent start at m = 1/2, nu = 2 ln 2: alpha = beta = 1 + ln 2 (physical std ~0.48).
            head.weight.mul_(0.01)
            head.bias.zero_()
            head.bias[action_dim:] = float(np.log(2.0 * LOG_2))
        self.log_kl_multiplier = nn.Parameter(torch.tensor(float(np.log(args.kl_start))))

    def forward(self, observations):
        return policy_logits(self.net, observations)


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


def lambda_returns(gae_fn, rewards, next_values, terminations, truncations, gamma, lam):
    """REPPO's GVE target G_t = r_t + gamma [(1 - lam) Q'_t + lam G_{t+1}] via the shared GAE kernel.

    Q'_t = E_{a'~pi} Q(s_{t+1}, a'). Boundaries bootstrap from Q'_t alone; terminations do not
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
    if args.num_envs % 2 or args.minibatch_size % 2:
        raise ValueError("num_envs and minibatch_size must be even (two environment folds)")
    if args.spread_critic not in CRITIC_ROLES:
        raise ValueError(f"spread_critic must be one of {sorted(CRITIC_ROLES)}")
    if args.aux_horizon <= 0 or args.aux_horizon >= args.num_steps:
        raise ValueError("aux_horizon must be in [1, num_steps)")
    if args.target_refresh <= 0 or args.critic_updates % args.target_refresh or args.bootstrap_samples <= 0:
        raise ValueError("target_refresh must divide critic_updates; bootstrap_samples positive")
    args.snapshot_at = sorted(int(float(step)) for step in args.snapshot_steps.split(",") if step.strip())
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
            raise ValueError("the Beta policy's 2 x - 1 action map requires a [-1, 1] Box action space")
        obs_shape = envs.single_observation_space.shape
        observation_dim, action_dim = int(np.prod(obs_shape)), int(np.prod(space.shape))
        roles = CRITIC_ROLES[args.spread_critic]
        E = 1 + max(max(fold_roles) for fold_roles in roles)  # critics in use
        actor = Actor(observation_dim, action_dim, args).to(device)
        critics = nn.ModuleList([Critic(observation_dim, action_dim, args, device) for _ in range(E)]).to(device)
        # Frozen per-iteration copies: the bootstrap targets, the KL anchor, and critic 0 one iteration older
        # than its target (the diagnostics compare their action gradients).
        target_critics = copy.deepcopy(critics).requires_grad_(False)
        previous_critic = copy.deepcopy(critics[0]).requires_grad_(False)
        # In-iteration refreshes move the targets, so the last iteration's final critic 0 is kept separately.
        final_critic = copy.deepcopy(critics[0]).requires_grad_(False)
        old_actor = copy.deepcopy(actor.net).requires_grad_(False)
        actor_network = list(actor.net.parameters())
        dual_parameters = [actor.log_kl_multiplier]
        actor_optimizer = torch.optim.AdamW(
            [{"params": actor_network},
             {"params": dual_parameters, "lr": args.dual_step_per_iteration / args.actor_updates,
              "weight_decay": 0.0}],
            lr=args.learning_rate, fused=True)
        member_parameters = [list(member.parameters()) for member in critics]
        critic_parameters = list(critics.parameters())
        critic_optimizer = torch.optim.AdamW([{"params": params} for params in member_parameters],
                                             lr=args.learning_rate, fused=True)
        actor_parameters = actor_network + dual_parameters
        B, M, N, T = args.batch_size, args.minibatch_size, args.num_envs, args.num_steps

        # Environment folds as (first env, env count); whole trajectories stay in one fold.
        env_folds = ((0, N // 2), (N // 2, N // 2))
        critic_envs = env_folds if args.spread_critic == "crossfold" else ((0, N),) * E

        # GPU copy of the fresh rollout (raw observations, rows in the rollout's (step, env) order, so a
        # row's environment is row % num_envs).
        f32 = dict(device=device, dtype=torch.float32)
        rollout_obs = torch.zeros((B, observation_dim), **f32)
        rollout_next_obs = torch.zeros((B, observation_dim), **f32)
        rollout_actions = torch.zeros((B, action_dim), **f32)
        rollout_rewards = torch.zeros(B, **f32)
        rollout_continuing = torch.zeros(B, **f32)
        rollout_truncated = torch.zeros(B, **f32)
        rollout_cuts = torch.zeros(B, **f32)
        # Per critic: its lambda-returns and f'(s, a) under its target (recomputed at each refresh, on its rows).
        rollout_targets = torch.zeros((E, B), **f32)
        target_features = torch.zeros((E, B, args.critic_hidden_dim), **f32)
        # rollout_chain[k - 1, row]: row's episode continues k steps within the rollout, so row + k N holds
        # s_{t+k}, a_{t+k}, r_{t+k}.
        H = args.aux_horizon
        rollout_chain = torch.zeros((H, B), **f32)
        diagnostic_targets = torch.zeros(B, **f32)
        time_shape = (T, N)
        rollout_ends = torch.zeros(time_shape, **f32)
        rollout_ends[-1] = 1.0
        rollout_rows = torch.arange(B, device=device)
        steps = torch.arange(T, device=device)[:, None]
        # All rows of each critic's environments, in (step, env) order.
        critic_rows = [(steps * N + torch.arange(start, start + count, device=device)).flatten()
                       for start, count in critic_envs]
        obs_mean = torch.zeros((N, observation_dim), **f32)
        obs_inv_std = torch.ones((N, observation_dim), **f32)
        critic_sums = torch.zeros((E, len(CRITIC_METRICS)), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(1 + E, **f32)
        diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed)

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def observations(rows):
            return normalize(rollout_obs[rows], rows)

        def next_observations(rows):
            return normalize(rollout_next_obs[rows], rows)

        def sample_rows(start, count, size):
            """Uniform rows of envs [start, start + count) from the default CUDA generator (graph-safe)."""
            step = (torch.rand(size, device=device) * T).long().clamp_max(T - 1)
            env = (torch.rand(size, device=device) * count).long().clamp_max(count - 1)
            return step * N + start + env

        def bootstrap_values(member, rows):
            """E_{a'~pi} Q_target(s', a') of one critic's own target over K current-policy actions."""
            next_obs = next_observations(rows)
            alpha, beta = beta_concentrations(*actor(next_obs))
            samples = args.bootstrap_samples
            next_actions = beta_actions(alpha.expand(samples, -1, -1), beta.expand(samples, -1, -1),
                                        reparameterized=False)
            return target_critics[member].value(next_obs.expand(samples, -1, -1), next_actions).mean(0)

        def encode_targets(member):
            """f'(s, a) for the critic's rollout rows under its (just refreshed) target."""
            rows = critic_rows[member]
            target_features[member].index_copy_(
                0, rows, target_critics[member].features(observations(rows), rollout_actions[rows]))

        def critic_loss(member, rows):
            critic = critics[member]
            obs, actions, targets = observations(rows), rollout_actions[rows], rollout_targets[member, rows]
            # Successor rows t + k (k = 1..H), clamped in range; invalid ones are masked by the chain. Valid
            # ones stay in the row's environment, hence in its fold (whose target features are written).
            later = torch.minimum(rows + N * torch.arange(1, H + 1, device=device)[:, None], rollout_rows[-1])
            chain = rollout_chain[:, rows]
            later_actions = rollout_actions[later]
            with torch.no_grad():
                later_features = target_features[member][later]
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

        # Actor rows are half fold 0, half fold 1. Each critic evaluates all its (role, fold) row blocks in one
        # forward: M rows per critic in every mode (crossfold: Q_0 sees fold-0 modes and fold-1 spreads).
        critic_blocks = [[] for _ in range(E)]
        for fold, fold_roles in enumerate(roles):
            for role, member in enumerate(fold_roles):
                critic_blocks[member].append((role, fold))

        def role_values(obs, mode_actions, spread_actions):
            """(Q_mode, Q_spread) per actor row under the --spread-critic assignment."""
            obs_folds = obs.chunk(2)
            action_folds = (mode_actions.chunk(2), spread_actions.chunk(2))
            values = {}
            for member, blocks in enumerate(critic_blocks):
                q = critics[member].value(torch.cat([obs_folds[fold] for _, fold in blocks]),
                                          torch.cat([action_folds[role][fold] for role, fold in blocks]))
                values.update(zip(blocks, q.chunk(len(blocks))))
            return tuple(torch.cat((values[role, 0], values[role, 1])) for role in (0, 1))

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_alpha, old_beta = beta_concentrations(*policy_logits(old_actor, obs))
            z_mode, z_precision = actor(obs)
            alpha, beta = beta_concentrations(z_mode, z_precision)
            # Mode and spread each get the pathwise gradient of their own critic, at independent samples.
            mode_actions = beta_actions(*beta_concentrations(z_mode, z_precision.detach()), reparameterized=True)
            spread_actions = beta_actions(*beta_concentrations(z_mode.detach(), z_precision), reparameterized=True)
            q_mode, q_spread = role_values(obs, mode_actions, spread_actions)
            kl_multiplier = actor.log_kl_multiplier.exp()
            objective = -(q_mode + q_spread)
            kl = beta_kl(old_alpha, old_beta, alpha, beta)
            inside = kl < args.kl_bound
            # Outside the trust region a state only receives the pull back toward the old policy.
            policy_loss = torch.where(inside, objective, kl * kl_multiplier.detach()).mean()
            multiplier_loss = -kl_multiplier * (kl.mean() - args.kl_bound).detach()
            metrics = torch.stack((objective.mean(), q_mode.mean(), q_spread.mean(),
                                   physical_entropy(alpha, beta).mean(), z_precision.exp().mean(),
                                   torch.tanh(0.5 * z_mode).abs().mean(), beta_std(alpha, beta).mean(), kl.mean(),
                                   1.0 - inside.float().mean(), kl_multiplier,
                                   (mode_actions.abs() > 0.99).float().mean())).detach()
            return policy_loss + multiplier_loss, metrics

        def action_gradient(model, obs, actions):
            actions = actions.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(model.value(obs, actions).sum(), actions)
            return grad

        def spread_gradients(rows, obs, z_mode, z_precision):
            """d E_{a~pi} Q(s, a) / d log nu per state (summed over dimensions) under each row's mode and spread
            critic, both at the same Beta samples (the spread's pathwise gradient; positive raises nu, i.e. narrows)."""
            samples = args.diagnostic_samples
            z_precision = z_precision.detach().requires_grad_(True)
            alpha, beta = beta_concentrations(z_mode.detach().expand(samples, -1, -1),
                                              z_precision.expand(samples, -1, -1))
            actions = beta_actions(alpha, beta, reparameterized=True)
            obs = obs.expand(samples, -1, -1)
            grads = torch.stack([
                torch.autograd.grad(target_critics[member].value(obs, actions).mean(0).sum(), z_precision,
                                    retain_graph=True)[0].sum(-1)
                for member in range(E)])
            member_roles = torch.tensor(roles, device=device)[(rows % N >= N // 2).long()]  # (rows, role)
            by_role = grads.gather(0, member_roles.T)
            return by_role[0], by_role[1]

        def critic_diagnostics(rows):
            """Is the critics' action dependence signal? Measured on fresh (held-out) rollout rows.

            dqda_cos_consecutive  cosine of critic 0's dQ/da at the policy mode between its target and the
                                  critic one iteration older: near 1 means the actor climbs a stable surface.
            dqda_cos_critics      the same cosine between the two critics (1 when there is one).
            advantage_corr        correlation of Q(s, a) - E_pi Q(s, .) with G_t - E_pi Q(s, .) on actions
                                  critic 0 has not been trained on.
            spread_grad_*         the spread's pathwise gradient under each row's mode critic (what a
                                  single-critic REPPO would use) and under its spread critic (what this uses).
            """
            obs = observations(rows)
            z_mode, z_precision = actor(obs)
            alpha, beta = beta_concentrations(z_mode, z_precision)
            mode_action = torch.tanh(0.5 * z_mode)
            with torch.enable_grad():
                current = action_gradient(target_critics[0], obs, mode_action)
                older = action_gradient(previous_critic, obs, mode_action)
                other = action_gradient(target_critics[-1], obs, mode_action)
                spread_mode, spread_spread = spread_gradients(rows, obs, z_mode, z_precision)
            consecutive = F.cosine_similarity(current, older, dim=-1, eps=1e-12).mean()
            across = F.cosine_similarity(current, other, dim=-1, eps=1e-12).mean()
            samples = args.diagnostic_samples
            sampled = beta_actions(alpha.expand(samples, -1, -1), beta.expand(samples, -1, -1),
                                   reparameterized=False)
            baseline = target_critics[0].value(obs.expand(samples, -1, -1), sampled).mean(0)
            critic_advantage = target_critics[0].value(obs, rollout_actions[rows]) - baseline
            return_advantage = diagnostic_targets[rows] - baseline
            centered_c = critic_advantage - critic_advantage.mean()
            centered_r = return_advantage - return_advantage.mean()
            correlation = (centered_c * centered_r).mean() / (centered_c.std() * centered_r.std() + 1e-12)
            return torch.stack((consecutive, across, current.norm(dim=-1).mean(), correlation,
                                centered_c.std() / (centered_r.std() + 1e-12), spread_mode.mean(),
                                spread_spread.mean(), (spread_spread > 0.0).float().mean()))

        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16)

        def per_member(fn):
            """fn bound to each critic member; one compiled graph per member."""
            def bind(member):
                return lambda *inputs: fn(member, *inputs)
            bound = [bind(member) for member in range(E)]
            return [graph_compile(variant) for variant in bound] if args.compile else bound

        encode_targets = per_member(encode_targets)
        bootstrap_values = per_member(bootstrap_values)
        critic_loss = per_member(critic_loss)
        if args.compile:
            actor_loss = graph_compile(actor_loss)

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            with autocast():
                results = [critic_loss[member](sample_rows(*critic_envs[member], M)) for member in range(E)]
            sum(loss for loss, _ in results).backward()
            for member, (_, metrics) in enumerate(results):
                grad_sums[1 + member] += nn.utils.clip_grad_norm_(member_parameters[member], args.max_grad_norm,
                                                                  foreach=True)
                critic_sums[member].add_(metrics)
            critic_optimizer.step()

        def actor_step():
            actor_optimizer.zero_grad(set_to_none=False)
            rows = torch.cat([sample_rows(start, count, M // 2) for start, count in env_folds])
            with autocast():
                loss, metrics = actor_loss(rows)
            # The critics are only differentiated through their action inputs.
            loss.backward(inputs=actor_parameters)
            grad_sums[0] += nn.utils.clip_grad_norm_(actor_network, args.max_grad_norm, foreach=True)
            actor_optimizer.step()
            actor_sums.add_(metrics)

        update_steps = None  # captured lazily, once the first rollout populated the buffers
        # One Triton kernel; a cudagraph-tree wrapper would fault the manually captured update graphs.
        gae_fn = get_gae_fn(compiled=args.compile, mode="default", explicit_next_values=True)
        critic_per_actor = args.critic_updates // args.actor_updates
        critic_tags = ["losses"] + [f"losses_{member}" for member in range(1, E)]  # critic 0 keeps v36's tags

        def refresh_targets(member, lam, rows, out):
            """Lambda-returns of the given rows (whole env columns) bootstrapped from one critic's target.

            Cut chains (truncation, rollout end) bootstrap from Q' alone; terminations bootstrap nothing.
            """
            grid_shape = (T, rows.numel() // T)
            continuing = rollout_continuing[rows].view(grid_shape)
            returns = lambda_returns(gae_fn, rollout_rewards[rows].view(grid_shape),
                                     bootstrap_values[member](rows).view(grid_shape), 1.0 - continuing,
                                     rollout_cuts[rows].view(grid_shape), args.gamma, lam)
            out.index_copy_(0, rows, returns.flatten())

        host_actor = make_host_mirror(actor.net, args.num_envs)
        writer.add_text("host_actor", f"fused={host_actor.fused} fallback={host_actor.fallback_reason}")
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, action_dim, space.low.reshape(-1).astype(np.float32),
                                           space.high.reshape(-1).astype(np.float32), epsilon=NATIVE_EPSILON,
                                           parameterization="mode")
        writer.add_text("host_sampler", f"fused={sample_actions.fused} fallback={sample_actions.fallback_reason}")

        def act(observations):
            """Stochastic Beta draw; the sampler's permanent buffer, overwritten by the next call."""
            _, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return action

        eval_actor = make_host_mirror(actor.net, 1) if args.eval_interval else None
        eval_env = gym.make(args.env_id) if args.eval_interval else None
        if eval_env is not None:
            resources.callback(eval_env.close)

        def evaluate(episode):
            """One mode-action episode, observations normalized with environment 0's statistics."""
            eval_actor.refresh()
            mean = obs_norm.means[0]
            inv_std = 1.0 / np.sqrt(obs_norm.variances[0] + obs_norm.epsilon)
            raw, _ = eval_env.reset(seed=args.seed + 10_000 + episode)
            total, saturated, steps_taken, done = 0.0, 0.0, 0, False
            while not done:
                normalized = np.clip((raw - mean) * inv_std, -args.obs_clip, args.obs_clip).astype(np.float32)
                # 2 m - 1 = 2 sigmoid(z_m) - 1 = tanh(z_m / 2).
                action = np.tanh(0.5 * eval_actor(normalized[None])[0, :action_dim])
                raw, reward, terminated, truncated, _ = eval_env.step(action)
                total += float(reward)
                saturated += float((np.abs(action) > 0.99).mean())
                steps_taken += 1
                done = terminated or truncated
            return total, saturated / steps_taken

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
                cuts = torch.maximum(rollout_truncated.view(time_shape), rollout_ends)
                rollout_cuts.copy_(cuts.flatten())
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(previous_critic.parameters()), list(final_critic.parameters()))
                torch._foreach_copy_(list(final_critic.parameters()), member_parameters[0])
                torch._foreach_copy_(list(target_critics.parameters()), critic_parameters)
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)
                # The episode survives step t -> t + 1 inside the rollout unless it terminated, truncated, or
                # the rollout ended at t.
                survives = rollout_continuing.view(time_shape) * (1.0 - cuts)
                chain = torch.ones_like(survives)
                for k in range(H):
                    chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
                    rollout_chain[k].copy_(chain.flatten())

            with timer.span("targets"), torch.no_grad():
                refresh_targets(0, args.diagnostic_lambda, rollout_rows, diagnostic_targets)

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
                                torch._foreach_copy_(list(target_critics.parameters()), critic_parameters)
                            for member in range(E):
                                refresh_targets(member, args.gve_lambda, critic_rows[member], rollout_targets[member])
                                encode_targets[member]()
                    critic_update()
                    if update % critic_per_actor == 0:
                        actor_update()

            if profiler is not None:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                with open(f"runs/{run_name}/update_profile.txt", "w") as handle:
                    handle.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40))
            logged = gather_metrics({
                **{f"{critic_tags[member]}/{name}": critic_sums[member, i] / args.critic_updates
                   for member in range(E) for i, name in enumerate(CRITIC_METRICS)},
                **{f"policy/{name}": actor_sums[i] / args.actor_updates for i, name in enumerate(ACTOR_METRICS)},
                **{f"diagnostics/{name}": diagnostics[i] for i, name in enumerate(DIAGNOSTIC_METRICS)},
                "grad/actor_preclip_norm": grad_sums[0] / args.actor_updates,
                **{f"grad/{tag.replace('losses', 'critic')}_preclip_norm": grad_sums[1 + member] / args.critic_updates
                   for member, tag in enumerate(critic_tags)},
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
                # Everything the oracle needs to rebuild the acting policy and the critics that judge it.
                torch.save({"actor_net": actor.net.state_dict(), "critic": critics[0].state_dict(),
                            "critics": critics.state_dict(), "obs_means": obs_norm.means.copy(),
                            "obs_variances": obs_norm.variances.copy(), "obs_epsilon": obs_norm.epsilon,
                            "global_step": global_step, "iteration": iteration, "args": vars(args),
                            "trainer_file": __file__},
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
