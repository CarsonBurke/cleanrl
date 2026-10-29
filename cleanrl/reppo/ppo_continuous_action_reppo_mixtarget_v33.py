# REPPO v33: v27 H8's critic fit to two estimators of the same Q^pi in every minibatch.
# v32 (v27 critic on 1M replay with 1-step targets) fell into HalfCheetah's ~2.3k local optimum by 1M, as has
# every short-horizon-target REPPO (v5, v10 with longer buffers, v12, v25 lambda 0.5); every fresh
# long-lambda variant escapes it. At 1M v32's critic valued its policy at 10.2 against ~18 implied by its
# returns: 1-step bootstraps propagate value too slowly while the actor commits to a gait. Conversely,
# v27's lambda-returns carry HalfCheetah's chaotic future-action noise, limiting late action ranking.
# The two targets have complementary errors, so fit both:
# - fresh half: the fresh rollout's soft lambda-returns (v25/v27: lambda 0.8, recomputed every
#   target_refresh updates) - long-horizon value immediately, on-policy.
# - replay half: y = r + gamma c E_{a'~pi}[Q'(s', a') - T log pi(a'|s')] on uniform rows of the last ~1M
#   transitions - off-policy-correct, free of future-action noise, many actions per state region.
# Both bootstrap from the Polyak twins' mean (one fixed point). Each twin keeps the H-step latent-roll and
# reward aux on all rows. Actor, KL gate and entropy dual are v27's.
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
CRITIC_METRICS = ("critic_ce", "aux_loss", "reward_loss", "q_mse", "support_clip_frac", "target_mean", "twin_gap")
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
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    minibatch_size: int = 2048
    """rows per critic and actor update"""
    bootstrap_samples: int = 4
    """current-policy actions averaged into each target's next-state soft value"""
    gve_lambda: float = 0.8
    """lambda of the fresh half's lambda-return targets"""
    target_refresh: int = 32
    """critic updates between recomputations of the fresh lambda-returns"""
    fresh_fraction: float = 0.5
    """share of each critic minibatch drawn from the fresh rollout (lambda-return targets)"""
    diagnostic_lambda: float = 0.95
    """lambda of the held-out diagnostic returns"""
    critic_updates: int = 512
    """critic updates per rollout, each on a uniform replay minibatch"""
    buffer_rollouts: int = 64
    """replay capacity in rollouts (64 x 16384 ~ 1M transitions)"""
    target_tau: float = 0.005
    """Polyak rate of the target critics per critic update"""
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
    """the replay capacity in transitions (computed in runtime)"""
    fresh_rows: int = 0
    """fresh rows per critic minibatch (computed in runtime)"""
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


class TwinCritic(nn.Module):
    """Two independent v27 critics."""

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        self.members = nn.ModuleList([Critic(observation_dim, action_dim, args, device) for _ in range(2)])

    def value(self, observations, actions):
        """The mean judgment the actor climbs."""
        first, second = (member.value(observations, actions) for member in self.members)
        return 0.5 * (first + second)


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
    if args.target_refresh <= 0 or args.critic_updates % args.target_refresh:
        raise ValueError("target_refresh must divide critic_updates")
    args.fresh_rows = int(round(args.fresh_fraction * args.minibatch_size))
    if not 0 < args.fresh_rows < args.minibatch_size:
        raise ValueError("fresh_fraction must leave both halves nonempty")
    if args.buffer_rollouts <= 0 or args.bootstrap_samples <= 0 or not 0.0 < args.target_tau <= 1.0:
        raise ValueError("buffer rollouts and bootstrap samples must be positive, target_tau in (0, 1]")
    if args.ent_start <= 0.0:
        raise ValueError("the initial entropy temperature must be positive")
    args.snapshot_at = sorted(int(float(step)) for step in args.snapshot_steps.split(",") if step.strip())
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.buffer_size = args.buffer_rollouts * args.batch_size
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
        critic = TwinCritic(observation_dim, action_dim, args, device).to(device)
        # Polyak targets and the critic one iteration older (diagnostics); old_actor is the KL anchor.
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        target_parameters = list(target_critic.parameters())
        previous_critic = copy.deepcopy(critic).requires_grad_(False)
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
        B, M, C, N, K = args.batch_size, args.minibatch_size, args.buffer_size, args.num_envs, args.bootstrap_samples

        # GPU replay of raw observations. Slots are whole rollouts in (step, env) order, so a row's
        # environment is row % num_envs and row + k N is the same environment k steps later.
        f32 = dict(device=device, dtype=torch.float32)
        rollout_obs = torch.zeros((C, observation_dim), **f32)
        rollout_next_obs = torch.zeros((C, observation_dim), **f32)
        rollout_actions = torch.zeros((C, action_dim), **f32)
        rollout_rewards = torch.zeros(C, **f32)
        rollout_continuing = torch.zeros(C, **f32)
        rollout_truncated = torch.zeros(B, **f32)
        # rollout_chain[k - 1, row]: row's episode continues k steps within its stored rollout, so row + k N
        # holds s_{t+k}, a_{t+k}, r_{t+k}.
        H = args.aux_horizon
        rollout_chain = torch.zeros((H, C), **f32)
        diagnostic_targets = torch.zeros(B, **f32)
        rollout_targets = torch.zeros(B, **f32)
        filled = torch.zeros((), **f32)
        fresh_start = torch.zeros((), dtype=torch.long, device=device)
        fresh_count = torch.full((), float(B), **f32)
        fresh_rows = torch.arange(B, device=device)
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

        def bootstrap_values(rows):
            """Expected soft value V'(s') = mean_k [Q'(s', a'_k) - T log pi(a'_k|s')] over K current-policy
            actions under the twins' mean (diagnostic lambda-returns only), and the entropy bonus
            T H^(s') = -T mean_k log pi(a'_k|s')."""
            next_obs = next_observations(rows)
            mean, std = actor(next_obs)
            next_actions, next_logprob = squashed_sample(mean.expand(K, -1, -1), std.expand(K, -1, -1))
            q = target_critic.value(next_obs.expand(K, -1, -1), next_actions)
            bonus = -actor.log_temperature.exp() * next_logprob.mean(0)
            return q.mean(0) + bonus, bonus

        def critic_loss(fresh_local, replay_rows):
            with torch.no_grad():
                next_value, _ = bootstrap_values(replay_rows)
                replay_targets = rollout_rewards[replay_rows] + args.gamma * rollout_continuing[replay_rows] * next_value
                rows = torch.cat((fresh_start + fresh_local, replay_rows))
                targets = torch.cat((rollout_targets[fresh_local], replay_targets))
            obs, actions = observations(rows), rollout_actions[rows]
            # Successor rows t + k (k = 1..H), clamped inside the row's stored rollout; invalid ones are
            # masked by the chain.
            slot_end = (rows // B) * B + (B - 1)
            later = torch.minimum(rows + N * torch.arange(1, H + 1, device=device)[:, None], slot_end)
            chain = rollout_chain[:, rows]
            later_obs, later_actions = observations(later), rollout_actions[later]
            # r_{t+k} from z_k, k = 0..H-1: z_0 always has its reward; z_k needs the chain to reach t + k.
            reward_mask = torch.cat((torch.ones_like(chain[:1]), chain[:-1]))
            reward_targets = rollout_rewards[torch.cat((rows[None], later[:-1]))]
            total = 0.0
            sums = torch.zeros(3, device=device)
            values = []
            for member, target_member in zip(critic.members, target_critic.members):
                with torch.no_grad():
                    later_features = target_member.features(later_obs, later_actions)
                    labels = member.hl.project(targets)
                features = member.features(obs, actions)
                logits = member.logits(features)
                ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
                latents = [features]
                for k in range(H):
                    latents.append(member.transition(latents[-1], later_actions[k]))
                latents = torch.stack(latents)
                prediction_error = (latents[1:] - later_features).square().mean(-1)
                aux = (chain * prediction_error).sum() / chain.sum().clamp_min(1.0)
                reward_error = (member.reward(latents[:-1]) - reward_targets).square()
                reward_loss = (reward_mask * reward_error).sum() / reward_mask.sum()
                total = total + ce + args.aux_loss_coef * aux + args.reward_loss_coef * reward_loss
                sums = sums + torch.stack((ce, aux, reward_loss)).detach()
                values.append(member.hl.to_scalar(logits.detach()))
            q_mse = (0.5 * (values[0] + values[1]) - targets).square().mean()
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            twin_gap = (values[0] - values[1]).abs().mean()
            return total, torch.cat((0.5 * sums, torch.stack((q_mse, support_clip, targets.mean(), twin_gap))))

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

            dqda_cos_consecutive  cosine of dQ/da at the policy mean between the live critic (not yet trained
                                  on this rollout) and the one an iteration before it: near 1 means
                                  the actor climbs a stable surface, near 0 that one iteration of fitting
                                  rewrites its direction.
            advantage_corr        correlation of Q(s, a) - E_pi Q(s, .) with G_t - E_pi Q(s, .) on actions
                                  the critic has not been trained on.
            """
            replay_rows = fresh_start + rows
            obs = observations(replay_rows)
            mean, std = actor(obs)
            deterministic = torch.tanh(mean)
            with torch.enable_grad():
                current = action_gradient(critic, obs, deterministic)
                older = action_gradient(previous_critic, obs, deterministic)
            cosine = nn.functional.cosine_similarity(current, older, dim=-1, eps=1e-12).mean()
            samples = args.diagnostic_samples
            noise = torch.randn((samples,) + mean.shape, device=device, generator=diagnostic_generator)
            sampled = torch.tanh(mean + std * noise)
            baseline = critic.value(obs.expand(samples, -1, -1), sampled).mean(0)
            critic_advantage = critic.value(obs, rollout_actions[replay_rows]) - baseline
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

        def fresh_uniform_rows():
            return torch.minimum((torch.rand(args.fresh_rows, device=device) * B).long(), rollout_rows[-1])

        def replay_uniform_rows():
            rows = (torch.rand(M - args.fresh_rows, device=device) * filled).long()
            return torch.minimum(rows, filled.long() - 1)

        def uniform_rows(count):
            """Uniform rows below count from the default CUDA generator (graph-safe, fresh per replay)."""
            return torch.minimum((torch.rand(M, device=device) * count).long(), count.long() - 1)

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            loss, metrics = critic_loss(fresh_uniform_rows(), replay_uniform_rows())
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            torch._foreach_lerp_(target_parameters, critic_parameters, args.target_tau)
            critic_sums.add_(metrics)

        def actor_step():
            actor_optimizer.zero_grad(set_to_none=False)
            # The actor trains on the fresh rollout's states (its KL anchor is the behavior policy).
            loss, metrics = actor_loss(fresh_start + uniform_rows(fresh_count))
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
                slot = ((iteration - 1) % args.buffer_rollouts) * B
                rows = slice(slot, slot + B)
                rollout_obs[rows].copy_(batch.fields["raw_obs"].flatten(0, 1))
                rollout_next_obs[rows].copy_(batch.fields["raw_next_obs"].flatten(0, 1))
                rollout_actions[rows].copy_(batch.fields["actions"].flatten(0, 1))
                rollout_rewards[rows].copy_(batch.rewards.flatten())
                rollout_continuing[rows].copy_(1.0 - batch.terminations.flatten())
                rollout_truncated.copy_(batch.truncations.flatten())
                filled.fill_(float(min(iteration, args.buffer_rollouts) * B))
                fresh_start.fill_(slot)
                torch.add(rollout_rows, slot, out=fresh_rows)
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

                cuts = torch.maximum(rollout_truncated.view(time_shape), rollout_ends)
                continuing = rollout_continuing[rows].view(time_shape)
                # The episode survives step t -> t + 1 inside the rollout unless it terminated, truncated, or
                # the rollout ended at t.
                survives = continuing * (1.0 - cuts)
                chain = torch.ones_like(survives)
                for k in range(H):
                    chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
                    rollout_chain[k, rows].copy_(chain.flatten())

            def refresh_targets(lam, out):
                """Soft lambda-returns G_t = r_t + gamma [(1 - lam) V'_{t+1} + lam (G_{t+1} + T H^_{t+1})] on the
                fresh rollout (v27): the expected entropy bonus is folded into the reward where the chain
                continues; cut chains bootstrap from V' alone; terminations bootstrap nothing."""
                next_values, bonus = bootstrap_values(fresh_rows)
                rewards = (rollout_rewards[rows].view(time_shape)
                           + args.gamma * lam * continuing * (1.0 - cuts) * bonus.view(time_shape))
                out.copy_(lambda_returns(gae_fn, rewards, next_values.view(time_shape),
                                         1.0 - continuing, cuts, args.gamma, lam).flatten())

            with timer.span("targets"), torch.no_grad():
                refresh_targets(args.diagnostic_lambda, diagnostic_targets)

            with timer.span("diagnostics"), torch.no_grad():
                # The actor still equals the rollout's behavior policy and the critic has not seen these rows.
                diagnostic_rows = torch.randperm(B, device=device, generator=diagnostic_generator)[: args.diagnostic_rows]
                diagnostics = critic_diagnostics(diagnostic_rows)
                torch._foreach_copy_(list(previous_critic.parameters()), critic_parameters)

            with timer.span("update"):
                if update_steps is None:
                    if args.compile:
                        graphs = capture_steps([critic_step, actor_step],
                                               optimizers=[critic_optimizer, actor_optimizer],
                                               state_tensors=[critic_sums, actor_sums, grad_sums, *target_parameters])
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
                            refresh_targets(args.gve_lambda, rollout_targets)
                    critic_update()
                    if update % critic_per_actor == 0:
                        actor_update()

            logged = gather_metrics({
                **{f"losses/{name}": critic_sums[i] / args.critic_updates for i, name in enumerate(CRITIC_METRICS)},
                **{f"policy/{name}": actor_sums[i] / args.actor_updates for i, name in enumerate(ACTOR_METRICS)},
                **{f"diagnostics/{name}": diagnostics[i] for i, name in enumerate(DIAGNOSTIC_METRICS)},
                "grad/actor_preclip_norm": grad_sums[0] / args.actor_updates,
                "grad/critic_preclip_norm": grad_sums[1] / args.critic_updates,
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
                torch.save({"actor_net": actor.net.state_dict(), "critic": critic.state_dict(), "trainer_file": __file__,
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
