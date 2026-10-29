# REPPO v28: v25 (in-iteration target refresh, expected soft bootstrap; lambda 0.8) with a latent-Q critic
# (port of td7/latentq's pre-action critic, which beat fast TD7 at 1M) and a deterministic actor ascent.
# The oracle showed the pointwise dQ/da REPPO's actor climbs is only ~0.25-cosine with the truth, and no
# change of target (v25), representation (v27) or head split (v26) moved the critic's held-out advantage
# fidelity. But the actor's objective is the noise-smoothed J(mu, sigma) = E_eps Q(s, tanh(mu + sigma eps)),
# so learn that directly and supervise its gradient with the realized noise:
# - Critic Q~(s, m, sigma, a), m = tanh(mu), queried at two points per transition:
#   center (m, sigma, m) = J(s, mu, sigma). Its forward-mode (jvp) tangent along the realized noise,
#   D = (1 - m^2) sigma eps . dQ~/da + (sigma / 2)(eps^2 - 1) . dQ~/dsigma, must explain the noise-
#   correlated part of the target: the L2 projection of y on the Hermite basis {eps_i, eps_i^2 - 1} has
#   coefficients sigma_i dJ/dmu_i and (sigma_i / 2) dJ/dsigma_i (Stein), exactly, so both gradients the
#   actor consumes are regressed against the sampled noise instead of inferred from a pointwise fit.
#   Point (a, 0, a) = Q(s, a) keeps full action coverage (diagnostics, aux, the sigma -> 0 boundary).
#   The center's value takes the HL-Gauss CE on y - D (D detached); its tangent a Gaussian likelihood
#   on y - J (J detached), scaled by the residual variance so it is unit-free like the CE.
# - Actor: ascend Q~(s, sg[m], sigma, m) deterministically (no reparameterized sample in the value
#   term): mu's gradient enters only through the Hermite-supervised action slot, sigma's through the
#   supervised sigma slot. Entropy dual and KL trust region unchanged.
# - Bootstrap: Q~'(s', m', sigma', m') is E_pi Q' directly, plus the K-sample entropy bonus.
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
CRITIC_METRICS = ("critic_ce", "aux_loss", "q_mse", "support_clip_frac", "target_mean", "hermite_r2",
                  "tangent_std", "smoothing_gap")
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
    gve_lambda: float = 0.8
    """lambda of the lambda-return critic target"""
    target_refresh: int = 16
    """critic updates between recomputations of the targets from the live critic"""
    bootstrap_samples: int = 4
    """current-policy actions in each bootstrap's entropy-bonus estimate"""
    tangent_coef: float = 1.0
    """weight of the center query's Hermite tangent likelihood"""
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
    """weight of the next-state feature prediction loss"""

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
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def normed_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), nn.RMSNorm(out_dim, eps=1e-5), nn.SiLU()]


class PrimitiveRMSNorm(nn.RMSNorm):
    """nn.RMSNorm in primitive ops: the fused CUDA rms_norm has no forward-mode (jvp) rule under
    torch.compile (fake-tensor _fw_primal assert). Same parameters and math; Inductor fuses it anyway."""

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def critic_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), PrimitiveRMSNorm(out_dim, eps=1e-5), nn.SiLU()]


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
    """Latent Q~(s, m, sigma, a): encoder -> {categorical value head, next-feature predictor}."""

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        width = args.critic_hidden_dim
        self.encoder = nn.Sequential(*critic_layer(observation_dim + 3 * action_dim, width), nn.Linear(width, width))
        self.value_head = nn.Sequential(nn.SiLU(), *critic_layer(width, width), nn.Linear(width, args.num_bins))
        self.predictor = nn.Sequential(nn.SiLU(), *critic_layer(width, width), nn.Linear(width, width))
        self.hl = HLGaussConfig(v_min=args.v_min, v_max=args.v_max, num_bins=args.num_bins).build(device)
        self.zero_dist = nn.Parameter(self.hl.project(torch.zeros((), device=device)))

    def features(self, observations, center, sigma, actions):
        return self.encoder(torch.cat((observations, center, sigma, actions), dim=-1))

    def logits(self, features):
        return self.value_head(features) + ZERO_DIST_SCALE * self.zero_dist

    def latent_value(self, observations, center, sigma, actions):
        return self.hl.to_scalar(self.logits(self.features(observations, center, sigma, actions)))

    def point_features(self, observations, actions):
        return self.features(observations, actions, torch.zeros_like(actions), actions)

    def value(self, observations, actions):
        """Pointwise Q(s, a) = Q~(s, a, 0, a)."""
        return self.hl.to_scalar(self.logits(self.point_features(observations, actions)))

    def policy_value(self, observations, mean, std):
        """J = E_eps Q(s, tanh(mean + std eps)) = Q~(s, tanh mean, std, tanh mean). The center slot is
        context only (detached): its derivative is not identified by one policy's data, while the
        action and sigma slots' derivatives are Hermite-supervised."""
        center = torch.tanh(mean)
        return self.latent_value(observations, center.detach(), std, center)


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs."""

    def __init__(self, num_envs, action_dim):
        shape = (num_envs, action_dim)
        self.mean = np.empty(shape, dtype=np.float32)
        self.std = np.empty(shape, dtype=np.float32)
        self.noise = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, outputs, rng):
        """Samples into the permanent buffers; mean, std and noise are kept for the latent critic."""
        mean, log_std = np.split(outputs, 2, axis=-1)
        np.copyto(self.mean, mean)
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


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates) <= 0:
        raise ValueError("environment, rollout, minibatch and update counts must be positive")
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
        B, M, N = args.batch_size, args.minibatch_size, args.num_envs

        # GPU copy of the fresh rollout (raw observations, rows in the rollout's (step, env) order, so a
        # row's environment is row % num_envs).
        f32 = dict(device=device, dtype=torch.float32)
        rollout_obs = torch.zeros((B, observation_dim), **f32)
        rollout_next_obs = torch.zeros((B, observation_dim), **f32)
        rollout_actions = torch.zeros((B, action_dim), **f32)
        # Behavior policy's pre-squash mean, std and the realized noise (action = tanh(mean + std noise)).
        rollout_mean = torch.zeros((B, action_dim), **f32)
        rollout_std = torch.zeros((B, action_dim), **f32)
        rollout_noise = torch.zeros((B, action_dim), **f32)
        rollout_rewards = torch.zeros(B, **f32)
        rollout_continuing = torch.zeros(B, **f32)
        rollout_truncated = torch.zeros(B, **f32)
        rollout_targets = torch.zeros(B, **f32)
        diagnostic_targets = torch.zeros(B, **f32)
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
            """Expected soft value V'(s') = Q~_frozen(s', tanh mu', sigma', tanh mu') + T H^(s'), with the entropy
            bonus T H^(s') = -T mean_k log pi(a'_k|s') over K current-policy actions."""
            next_obs = next_observations(rows)
            mean, std = actor(next_obs)
            samples = args.bootstrap_samples
            next_logprob = squashed_sample(mean.expand(samples, -1, -1), std.expand(samples, -1, -1))[1]
            bonus = -actor.log_temperature.exp() * next_logprob.mean(0)
            return target_critic.policy_value(next_obs, mean, std) + bonus, bonus

        def critic_loss(rows):
            obs, next_obs = observations(rows), next_observations(rows)
            actions, continuing, targets = rollout_actions[rows], rollout_continuing[rows], rollout_targets[rows]
            center, sigma, noise = torch.tanh(rollout_mean[rows]), rollout_std[rows], rollout_noise[rows]
            # Hermite directions of the realized noise: first order in the action slot (through
            # d tanh(u)/du), second order in the sigma slot.
            action_tangent = (1.0 - center.square()) * sigma * noise
            sigma_tangent = 0.5 * sigma * (noise.square() - 1.0)
            with torch.no_grad():
                next_features = target_critic.point_features(next_obs, squashed_sample(*actor(next_obs))[0])
            point_features = critic.point_features(obs, actions)
            point_logits = critic.logits(point_features)
            # Center query (m, sigma, m) and its forward-mode derivative along the Hermite directions.
            q_center, tangent = torch.func.jvp(lambda a, sd: critic.latent_value(obs, center, sd, a),
                                               (center, sigma), (action_tangent, sigma_tangent))
            with torch.no_grad():
                labels = critic.hl.project(torch.cat((targets, targets - tangent)))
            center_logits = critic.logits(critic.features(obs, center, sigma, center))
            all_logits = torch.cat((point_logits, center_logits))
            ce = -(labels * all_logits.log_softmax(dim=-1)).sum(-1).mean()
            with torch.no_grad():
                residual = (targets - q_center).square().mean()
            tangent_nll = 0.5 * (q_center.detach() + tangent - targets).square().mean() / residual.clamp_min(1e-6)
            prediction_error = (critic.predictor(point_features) - next_features).square().mean(-1)
            aux = (continuing * prediction_error).mean()
            with torch.no_grad():
                point_q = critic.hl.to_scalar(point_logits)
                q_mse = (point_q - targets).square().mean()
                hermite_r2 = 1.0 - (targets - q_center - tangent).square().mean() / residual.clamp_min(1e-8)
                smoothing_gap = (point_q - q_center).mean()
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            loss = ce + args.tangent_coef * tangent_nll + args.aux_loss_coef * aux
            return loss, torch.stack((ce.detach(), aux.detach(), q_mse, support_clip, targets.mean(), hermite_r2,
                                      tangent.detach().std(), smoothing_gap))

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs)
            mean, std = actor(obs)
            actions, logprob = squashed_sample(mean, std)
            # Deterministic ascent of the smoothed value; the sample only estimates the entropy.
            q = critic.policy_value(obs, mean, std)
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
            """Pointwise dQ/da (the point query), comparable with v23-v27."""
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
            """Uniform fresh-rollout rows from the default CUDA generator (graph-safe)."""
            return torch.minimum((torch.rand(M, device=device) * B).long(), rollout_rows[-1])

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
                                   fields={"raw_obs": obs_shape, "raw_next_obs": obs_shape, "actions": (action_dim,),
                                       "policy_mean": (action_dim,), "policy_std": (action_dim,),
                                       "noise": (action_dim,)})
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
                                  raw_obs=raw_current, raw_next_obs=raw_next, actions=action,
                                  policy_mean=sample_actions.mean, policy_std=sample_actions.std,
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
                rollout_mean.copy_(batch.fields["policy_mean"].flatten(0, 1))
                rollout_std.copy_(batch.fields["policy_std"].flatten(0, 1))
                rollout_noise.copy_(batch.fields["noise"].flatten(0, 1))
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

            cuts = torch.maximum(rollout_truncated.view(time_shape), rollout_ends)

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
                torch.save({"actor_net": actor.net.state_dict(), "critic": critic.state_dict(),
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
