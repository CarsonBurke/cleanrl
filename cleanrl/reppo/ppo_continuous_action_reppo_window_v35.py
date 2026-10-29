# REPPO v35: v27 H8 with the policy's update period decoupled from the critic's data window.
# v27 couples them: one 16 envs x 1024-step rollout is both the critic's entire dataset and the policy's
# update period, so the policy changes ~490 times in 8M steps (PPO here: ~3900), taking 32 actor steps
# against a critic fit before any of them and pinning 30-37% of states at the KL bound, although the audited
# late updates improve real returns. More critic epochs on that data hurt (v7 cu256) and off-policy replay
# hurts (v32-v34): the critic wants a current-policy window of about this size, the actor wants to move more
# often. So:
# - Each iteration rolls out window_segment steps (default 128) per env; the critic fits the last
#   window_segments segments (default 8: v27's 16k transitions) kept as one time-contiguous block per env, so
#   v27's soft lambda-returns and H-step latent-roll chains run across segment boundaries unchanged.
# - Behavior policies inside one window differ by up to ~window_segments KL steps (v27: one); the lambda
#   branch uses those behavior actions uncorrected - the price of the fresher updates.
# - Per environment step the critic updates, actor updates, target refreshes and dual step budget equal
#   v27's; the policy (and the KL anchor) updates window_segments times more often, each actor step against
#   a critic refit on the newest data.
# Hypothesis: faster, fresher policy improvement at v27's compute, lifting both early speed and the plateau.
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
DIAGNOSTIC_METRICS = ("dqda_cos_consecutive", "dqda_norm", "advantage_corr_heldout", "advantage_std_ratio")


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
    num_steps: int = 128
    """steps per environment per iteration (one window segment)"""
    window_segments: int = 8
    """segments in the critic's time-contiguous window (8 x 128 = v27's 1024 steps per env)"""
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
    critic_updates: int = 16
    """critic updates per iteration, each on a uniform minibatch of the window (v27's rate per env step)"""
    actor_updates: int = 4
    """actor updates per iteration, interleaved evenly with the critic's (v27's rate per env step)"""
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
    """initial entropy temperature"""
    ent_target_mult: float = 0.5
    """target entropy is -ent_target_mult * action_dim"""
    kl_bound: float = 0.03
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
    dual_step_per_iteration: float = 0.01875
    """log-space step budget of each dual variable per iteration (v27's 0.15 per 16k transitions)"""
    obs_clip: float = 10.0
    """observation normalization clip (VectorObsNorm's)"""

    # Diagnostics
    diagnostic_rows: int = 2048
    """newest-segment (not yet trained on) rows of the critic diagnostics"""
    diagnostic_samples: int = 8
    """policy samples in the diagnostics' state baseline E_{a~pi} Q(s, a)"""
    eval_interval: int = 200
    """iterations between deterministic (mean-action) evaluation episodes; 0 disables"""
    log_interval: int = 8
    """iterations per logged interval (learner metrics are averaged over it; diagnostics run once)"""
    snapshot_steps: str = ""
    """comma-separated global steps (e.g. 4e6,12e6) after which actor, critic and normalization are saved"""

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
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    window_steps: int = 0
    """steps per environment in the critic's window (computed in runtime)"""
    window_size: int = 0
    """transitions in the critic's window (computed in runtime)"""


def normed_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), nn.RMSNorm(out_dim, eps=1e-5), nn.SiLU()]


def policy_moments(net, observations):
    """Pre-squash mean and standard deviation."""
    mean, log_std = net(observations).chunk(2, dim=-1)
    return mean, log_std.exp()


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
    if args.window_segments <= 0 or args.log_interval != args.window_segments:
        raise ValueError("window_segments must be positive; log_interval must equal it (diagnostic critics are one window apart)")
    if args.eval_interval % args.log_interval:
        raise ValueError("eval_interval must be a multiple of log_interval (evaluation runs at logged iterations)")
    args.window_steps = args.window_segments * args.num_steps
    if args.aux_horizon <= 0 or args.aux_horizon >= args.window_steps:
        raise ValueError("aux_horizon must be in [1, window_steps)")
    if args.target_refresh <= 0 or args.critic_updates % args.target_refresh or args.bootstrap_samples <= 0:
        raise ValueError("target_refresh must divide critic_updates; bootstrap_samples positive")
    if args.ent_start <= 0.0:
        raise ValueError("the initial entropy temperature must be positive")
    args.snapshot_at = sorted(int(float(step)) for step in args.snapshot_steps.split(",") if step.strip())
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.window_size = args.num_envs * args.window_steps
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
        actor_parameters = actor_network + dual_parameters
        target_entropy = -args.ent_target_mult * action_dim
        B, W, M, N = args.batch_size, args.window_size, args.minibatch_size, args.num_envs

        # GPU window of the last window_segments segments (raw observations), one time-contiguous block in
        # (step, env) order: a row's environment is row % num_envs and row + k N is its step t + k. Each
        # iteration shifts it by one segment (B rows) and appends the fresh segment at the end. Rows before
        # valid_start are not yet filled (zeros: they neither bootstrap nor continue into valid rows).
        f32 = dict(device=device, dtype=torch.float32)
        rollout_obs = torch.zeros((W, observation_dim), **f32)
        rollout_next_obs = torch.zeros((W, observation_dim), **f32)
        rollout_actions = torch.zeros((W, action_dim), **f32)
        rollout_rewards = torch.zeros(W, **f32)
        rollout_continuing = torch.zeros(W, **f32)
        rollout_truncated = torch.zeros(W, **f32)
        rollout_targets = torch.zeros(W, **f32)
        window_buffers = [rollout_obs, rollout_next_obs, rollout_actions, rollout_rewards, rollout_continuing,
                          rollout_truncated]
        # rollout_chain[k - 1, row]: row's episode continues k steps within the window, so row + k N holds
        # s_{t+k}, a_{t+k}, r_{t+k}.
        H = args.aux_horizon
        rollout_chain = torch.zeros((H, W), **f32)
        diagnostic_targets = torch.zeros(W, **f32)
        time_shape = (args.window_steps, N)
        rollout_ends = torch.zeros(time_shape, **f32)
        rollout_ends[-1] = 1.0
        rollout_rows = torch.arange(W, device=device)
        valid_start = torch.full((), W, dtype=torch.long, device=device)
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
            with torch.no_grad():
                later_features = target_critic.features(observations(later), later_actions)
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
            return torch.stack((cosine, current.norm(dim=-1).mean(), correlation,
                                centered_c.std() / (centered_r.std() + 1e-12)))

        if args.compile:
            bootstrap_values = graph_compile(bootstrap_values)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        def uniform_rows():
            """Uniform filled-window rows from the default CUDA generator (graph-safe)."""
            rows = valid_start + (torch.rand(M, device=device) * (W - valid_start)).long()
            return torch.minimum(rows, rollout_rows[-1])

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            loss, metrics = critic_loss(uniform_rows())
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            critic_sums.add_(metrics)

        def actor_step():
            actor_optimizer.zero_grad(set_to_none=False)
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
            log_start = (iteration - 1) % args.log_interval == 0
            log_end = iteration % args.log_interval == 0 or iteration == args.num_iterations
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
                # Shift the window one segment back in time and append the fresh segment.
                for buffer in window_buffers:
                    buffer.copy_(buffer.roll(-B, 0))
                rollout_obs[-B:].copy_(batch.fields["raw_obs"].flatten(0, 1))
                rollout_next_obs[-B:].copy_(batch.fields["raw_next_obs"].flatten(0, 1))
                rollout_actions[-B:].copy_(batch.fields["actions"].flatten(0, 1))
                rollout_rewards[-B:].copy_(batch.rewards.flatten())
                rollout_continuing[-B:].copy_(1.0 - batch.terminations.flatten())
                rollout_truncated[-B:].copy_(batch.truncations.flatten())
                valid_start.fill_(W - min(iteration, args.window_segments) * B)
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                if log_start:
                    # Diagnostic critics are one window (v27's rollout) apart.
                    torch._foreach_copy_(list(previous_critic.parameters()), list(final_critic.parameters()))
                    torch._foreach_copy_(list(final_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
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

            if log_start:
                with timer.span("targets"), torch.no_grad():
                    refresh_targets(args.diagnostic_lambda, diagnostic_targets)

                with timer.span("diagnostics"), torch.no_grad():
                    # The actor still equals the fresh segment's behavior policy, and the critic has not
                    # trained on that segment.
                    diagnostic_rows = W - B + torch.randperm(B, device=device, generator=diagnostic_generator)[: args.diagnostic_rows]
                    diagnostics = critic_diagnostics(diagnostic_rows)

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
                if log_start:
                    critic_sums.zero_()
                    actor_sums.zero_()
                    grad_sums.zero_()
                    interval_iterations = 0
                interval_iterations += 1
                for update in range(1, args.critic_updates + 1):
                    if (update - 1) % args.target_refresh == 0:
                        with torch.no_grad():
                            if update > 1:
                                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                            refresh_targets(args.gve_lambda, rollout_targets)
                    critic_update()
                    if update % critic_per_actor == 0:
                        actor_update()

            if log_end:
                critic_count = args.critic_updates * interval_iterations
                actor_count = args.actor_updates * interval_iterations
                logged = gather_metrics({
                    **{f"losses/{name}": critic_sums[i] / critic_count for i, name in enumerate(CRITIC_METRICS)},
                    **{f"policy/{name}": actor_sums[i] / actor_count for i, name in enumerate(ACTOR_METRICS)},
                    **{f"diagnostics/{name}": diagnostics[i] for i, name in enumerate(DIAGNOSTIC_METRICS)},
                    "grad/actor_preclip_norm": grad_sums[0] / actor_count,
                    "grad/critic_preclip_norm": grad_sums[1] / critic_count,
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
                            "obs_means": obs_norm.means.copy(), "obs_variances": obs_norm.variances.copy(),
                            "obs_epsilon": obs_norm.epsilon, "global_step": global_step, "iteration": iteration,
                            "temperature": float(actor.log_temperature.exp()), "args": vars(args)},
                           f"runs/{run_name}/snapshot_{global_step}.pt")
            if log_end:
                now = time.perf_counter()
                writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
                writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
                for phase, timing in timer.summary().items():
                    writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
                timer.reset()
                print(f"SPS: {int(global_step / (now - start_time))}")
                interval_start, interval_step = time.perf_counter(), global_step
    finally:
        resources.close()


if __name__ == "__main__":
    main()
