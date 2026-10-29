# REPPO v17: v16's distribution-critic GVE, with the actor back on REPPO's pathwise gradient through the
# point query Q(s, tanh u), u = mu + sigma eps (v16 ascended Q~(s, mu, sigma) directly).
# v16 lost to v7 cu128 (4992 vs 8013 @4M) with half its actor gradient norm. In TD7 the point query
# supervises the surface the actor climbs (sigma ~ 0.1, so Q~(s, mu, sigma) ~ Q(s, mu)); at REPPO's
# pre-tanh sigma ~ 0.35 the (mu, sigma) slope is identified only by the weak Hermite tangent
# (hermite_r2 ~ 0.05). The reparameterized point-query gradient already estimates grad Q~ without bias,
# from the densely supervised Q(s, a). Q~ keeps the job where it is on-support and exact: the
# bootstrap V^(s') = Q~'(s', mu(s'), sigma(s')) + T H_pi(s'), evaluated at the behavior latent it was
# trained on. Hypothesis: the exact distribution bootstrap lowers target variance over v7's one-sample
# Q'(s', a') without v16's actor-signal loss.
#
# v16 notes (unchanged): GVE for the distribution critic Q~(s, m, sigma) = E_eps Q(s, tanh(m + sigma eps)), after
# TD7 latentq dual_smooth_v3 (the configuration where Q~ beat TD7's Q(s, a)). Base: v7 at 128 critic
# updates per rollout, fresh rollout only. v12/v15 never trained Q~ on returns (it was a smoothed copy
# of the point critic); here the returns train Q~ itself and Q~ bootstraps itself:
#   G_t = r_t + gamma [ (1 - lam) V^_{t+1} + lam (G_{t+1} - T log pi(a_{t+1}|s_{t+1})) ]
#   V^(s') = Q~'(s', mu(s'), sigma(s')) + T H_pi(s')
# The bootstrap is the exact expected soft value of the current policy's distribution (REPPO: one
# sampled a'); the lambda-branch subtracts the entropy term of the action actually taken (REPPO reused a').
# Chains cut at terminations (G = r), truncations and rollout ends (G = r + gamma V^). H_pi is a K-sample
# reparameterized estimate (the tanh-Gaussian entropy has no closed form).
# Two queries per row regress onto G_t, one categorical (HL-Gauss) critic:
#   behavior  Q~(s, mu_b, sigma_b) + tau, tau = sigma_b eps d/dmu Q~ + sigma_b (eps^2 - 1)/2 d/dsigma Q~,
#             eps = (u - mu_b)/sigma_b the realized pre-tanh noise: the Hermite (Stein) projection that
#             identifies Q~'s slopes in (mu, sigma) from the actions actually taken. tau is an exact
#             forward-mode derivative. Loss: CE(Q~, G - sg(tau)) + Huber(tau - (G - sg(Q~))), whose gradient
#             matches TD7's joint Huber(Q~ + tau - G) with the categorical loss in place of the Q~ part.
#   point     Q~(s, u, 0) = Q(s, a) (TD7's degenerate latent), REPPO's action critic + aux loss.
# The behavior moments are the host sampler's own, so eps is exactly the drawn noise; the tangent
# residual is Huber (TD7's LAP), keeping it from dominating the clipped gradient while Q~ lags.
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
from cleanrl.shared.linear_scan import reverse_linear_scan
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
CRITIC_METRICS = ("point_ce", "dist_ce", "tangent_loss", "aux_loss", "q_mse", "hermite_r2", "tangent_std",
                  "point_dist_gap", "support_clip_frac", "target_mean")
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
    """lambda of the soft lambda-return critic target"""
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
    """lowest supported scaled soft return"""
    v_max: float = 250.0
    """highest supported scaled soft return"""
    aux_loss_coef: float = 1.0
    """weight of the next-state feature prediction loss"""
    tangent_coef: float = 1.0
    """weight of the Hermite tangent regression"""
    hermite_order: int = 2
    """1: tangent along the mean only; 2: also along the std"""
    entropy_samples: int = 16
    """reparameterized samples of the bootstrap's policy-entropy estimate"""

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
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def normed_layer(in_dim, out_dim):
    return [nn.Linear(in_dim, out_dim), nn.RMSNorm(out_dim, eps=1e-5), nn.SiLU()]


def squashed_sample(mean, std):
    """Reparameterized pre-tanh sample u (action tanh u) and the exact log-density of tanh u."""
    noise = torch.randn_like(mean)
    pre = mean + std * noise
    # log|d tanh(u)/du| = 2 (log 2 - u - softplus(-2u)), stable for large |u|.
    log_det = 2.0 * (LOG_2 - pre - F.softplus(-2.0 * pre))
    logprob = (-0.5 * noise.square() - std.log() - LOG_SQRT_2PI - log_det).sum(-1)
    return pre, logprob


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
    """Q~(s, m, sigma) over pre-tanh Gaussians: encoder -> {categorical value head, next-feature predictor}.

    The network reads (s, tanh m, sigma); sigma = 0 is REPPO's action critic Q(s, tanh m).
    """

    def __init__(self, observation_dim, action_dim, args, device):
        super().__init__()
        width = args.critic_hidden_dim
        self.encoder = nn.Sequential(*normed_layer(observation_dim + 2 * action_dim, width), nn.Linear(width, width))
        self.value_head = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, args.num_bins))
        self.predictor = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, width))
        self.hl = HLGaussConfig(v_min=args.v_min, v_max=args.v_max, num_bins=args.num_bins).build(device)
        self.zero_dist = nn.Parameter(self.hl.project(torch.zeros((), device=device)))

    def features(self, observations, locations, scales):
        return self.encoder(torch.cat((observations, torch.tanh(locations), scales), dim=-1))

    def logits(self, features):
        return self.value_head(features) + ZERO_DIST_SCALE * self.zero_dist

    def value(self, observations, locations, scales):
        return self.hl.to_scalar(self.logits(self.features(observations, locations, scales)))

    def value_and_tangent(self, observations, locations, scales, location_tangents, scale_tangents):
        """Q~, its directional derivative along (location_tangents, scale_tangents), the logits and the
        encoder features, by an exact forward-mode pass (graph-safe; differentiable in the parameters)."""
        actions = torch.tanh(locations)
        x = torch.cat((observations, actions, scales), dim=-1)
        t = torch.cat((torch.zeros_like(observations), (1.0 - actions.square()) * location_tangents,
                       scale_tangents), dim=-1)
        features = None
        for module in (*self.encoder, None, *self.value_head):
            if module is None:
                features = x
            elif isinstance(module, nn.Linear):
                x, t = module(x), t @ module.weight.T
            elif isinstance(module, nn.RMSNorm):
                inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + module.eps)
                x, t = (x * inv * module.weight,
                        module.weight * inv * (t - x * inv.square() * (x * t).mean(-1, keepdim=True)))
            elif isinstance(module, nn.SiLU):
                gate = torch.sigmoid(x)
                x, t = x * gate, t * gate * (1.0 + x * (1.0 - gate))
            else:
                raise TypeError(f"no tangent rule for {type(module).__name__}")
        logits = x + ZERO_DIST_SCALE * self.zero_dist
        probs = logits.softmax(-1)
        value = (probs * self.hl.support).sum(-1)
        tangent = (probs * (self.hl.support - value.unsqueeze(-1)) * t).sum(-1)
        return value, tangent, logits, features


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs; keeps the sampling moments
    and the pre-tanh sample, so the behavior noise (pre - mean) / std is exactly the drawn one."""

    def __init__(self, num_envs, action_dim, min_std):
        shape = (num_envs, action_dim)
        self.min_std = np.float32(min_std)
        self.mean = np.empty(shape, dtype=np.float32)
        self.std = np.empty(shape, dtype=np.float32)
        self.pre = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
        np.copyto(self.mean, mean)
        np.exp(log_std, out=self.std)
        self.std += self.min_std
        rng.standard_normal(self.pre.shape, dtype=np.float32, out=self.pre)
        self.pre *= self.std
        self.pre += self.mean
        np.tanh(self.pre, out=self.action)
        return self.action


def squashed_logprob(pre, mean, std):
    """log pi(tanh u | s) of a stored pre-tanh sample u under N(mean, std^2)."""
    log_det = 2.0 * (LOG_2 - pre - F.softplus(-2.0 * pre))
    return (-0.5 * ((pre - mean) / std).square() - std.log() - LOG_SQRT_2PI - log_det).sum(-1)


def shift_up(tensor):
    """Row t of the result is row t + 1 of a time-major tensor (zero past the end)."""
    return torch.cat((tensor[1:], torch.zeros_like(tensor[:1])))


def distribution_gve(rewards, soft_values, taken_logprob, continuing, chain_ends, temperature, gamma, lam):
    """G_t = r_t + gamma cont_t [(1 - lam c_t) V^_{t+1} + lam c_t (G_{t+1} - T log pi(a_{t+1}|s_{t+1}))] over
    time-major (T, N), c_t = 1 - chain_end_t; V^_t, the soft value at s_{t+1}, is row t of soft_values."""
    chained = continuing * (1.0 - chain_ends)
    coefficients = gamma * lam * chained
    inputs = (rewards + gamma * continuing * soft_values
              - coefficients * (soft_values + temperature * shift_up(taken_logprob)))
    return reverse_linear_scan(inputs, coefficients)


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates) <= 0:
        raise ValueError("environment, rollout, minibatch and update counts must be positive")
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.hermite_order not in (1, 2) or args.entropy_samples < 1:
        raise ValueError("hermite_order must be 1 or 2 and entropy_samples positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
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
        B, M, N = args.batch_size, args.minibatch_size, args.num_envs

        # The fresh rollout on the GPU as raw observations, (step, env)-flattened: a row's environment is
        # row % num_envs. Observations are re-normalized with one set of statistics per iteration.
        f32 = dict(device=device, dtype=torch.float32)
        rollout_obs = torch.zeros((B, observation_dim), **f32)
        rollout_next_obs = torch.zeros((B, observation_dim), **f32)
        rollout_pre_actions = torch.zeros((B, action_dim), **f32)
        # Behavior moments (the policy at iteration start) and each action's realized standardized noise.
        rollout_behavior_mean = torch.zeros((B, action_dim), **f32)
        rollout_behavior_std = torch.ones((B, action_dim), **f32)
        rollout_rewards = torch.zeros(B, **f32)
        rollout_continuing = torch.zeros(B, **f32)
        rollout_truncated = torch.zeros(B, **f32)
        rollout_targets = torch.zeros(B, **f32)
        # Every rollout ends in a bootstrap.
        time_shape = (args.num_steps, N)
        rollout_ends = torch.zeros(time_shape, **f32)
        rollout_ends[-1] = 1.0
        obs_mean = torch.zeros((N, observation_dim), **f32)
        obs_inv_std = torch.ones((N, observation_dim), **f32)
        critic_sums = torch.zeros(len(CRITIC_METRICS), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(2, **f32)

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def observations(rows):
            return normalize(rollout_obs[rows], rows)

        def next_observations(rows):
            return normalize(rollout_next_obs[rows], rows)

        def target_inputs():
            """log pi of each taken action and the soft value V^(s') = Q~'(s', mu, sigma) + T H_pi(s') of the
            current policy at every s'."""
            rows = torch.arange(B, device=device)
            obs, next_obs = observations(rows), next_observations(rows)
            mean, std = actor(obs)
            taken_logprob = squashed_logprob(rollout_pre_actions, mean, std)
            next_mean, next_std = actor(next_obs)
            _, sample_logprob = squashed_sample(next_mean.expand(args.entropy_samples, -1, -1),
                                                next_std.expand(args.entropy_samples, -1, -1))
            entropy = -sample_logprob.mean(0)
            return taken_logprob, target_critic.value(next_obs, next_mean, next_std), entropy

        def critic_loss(rows):
            obs, next_obs = observations(rows), next_observations(rows)
            pre, continuing, targets = rollout_pre_actions[rows], rollout_continuing[rows], rollout_targets[rows]
            mean_b, std_b = rollout_behavior_mean[rows], rollout_behavior_std[rows]
            with torch.no_grad():
                next_mean, next_std = actor(next_obs)
                next_pre = next_mean + next_std * torch.randn_like(next_std)
                next_features = target_critic.features(next_obs, next_pre, torch.zeros_like(next_pre))
                labels = critic.hl.project(targets)
                noise = (pre - mean_b) / std_b
                location_tangents = std_b * noise
                scale_tangents = (0.5 * std_b * (noise.square() - 1.0) if args.hermite_order == 2
                                  else torch.zeros_like(std_b))
            # Behavior (distribution) query with its Hermite tangent.
            q_dist, tangent, dist_logits, _ = critic.value_and_tangent(obs, mean_b, std_b, location_tangents,
                                                                         scale_tangents)
            dist_ce = critic.hl.loss(dist_logits, (targets - tangent).detach())
            # Huber, as TD7's LAP loss on the joint residual: bounded while Q~ lags the returns.
            tangent_loss = F.huber_loss(tangent, (targets - q_dist).detach(), delta=1.0)
            # Point query (degenerate latent): REPPO's Q(s, a) with the aux loss on its features.
            features = critic.features(obs, pre, torch.zeros_like(pre))
            logits = critic.logits(features)
            point_ce = -(labels * logits.log_softmax(dim=-1)).sum(-1).mean()
            prediction_error = (critic.predictor(features) - next_features).square().mean(-1)
            aux = (continuing * prediction_error).mean()
            loss = point_ce + dist_ce + args.tangent_coef * tangent_loss + args.aux_loss_coef * aux
            with torch.no_grad():
                q_point = critic.hl.to_scalar(logits)
                residual = (targets - q_dist).square().mean()
                hermite_r2 = 1.0 - (targets - q_dist - tangent).square().mean() / residual.clamp_min(1e-8)
                support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
                metrics = torch.stack((point_ce, dist_ce, tangent_loss, aux, (q_point - targets).square().mean(),
                                       hermite_r2, tangent.std(), (q_point - q_dist).mean(), support_clip,
                                       targets.mean()))
            return loss, metrics

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs, args.actor_min_std)
            mean, std = actor(obs)
            pre, logprob = squashed_sample(mean, std)
            q = critic.value(obs, pre, torch.zeros_like(pre))
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
            target_inputs = graph_compile(target_inputs)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        def uniform_rows():
            """Uniform fresh-rollout rows from the default CUDA generator (graph-safe, fresh per replay)."""
            return (torch.rand(M, device=device) * B).long().clamp_(max=B - 1)

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
            # The critic is only differentiated through its (mean, std) inputs.
            loss.backward(inputs=actor_parameters)
            grad_sums[0] += nn.utils.clip_grad_norm_(actor_network, args.max_grad_norm, foreach=True)
            actor_optimizer.step()
            actor_sums.add_(metrics)

        update_steps = None  # captured lazily, once the first rollout is stored
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
                                           "pre_actions": (action_dim,), "behavior_mean": (action_dim,),
                                           "behavior_std": (action_dim,)})
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
                                  raw_obs=raw_current, raw_next_obs=raw_next, pre_actions=sample_actions.pre,
                                  behavior_mean=sample_actions.mean, behavior_std=sample_actions.std)
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
                rollout_pre_actions.copy_(batch.fields["pre_actions"].flatten(0, 1))
                rollout_behavior_mean.copy_(batch.fields["behavior_mean"].flatten(0, 1))
                rollout_behavior_std.copy_(batch.fields["behavior_std"].flatten(0, 1))
                rollout_rewards.copy_(batch.rewards.flatten())
                rollout_continuing.copy_(1.0 - batch.terminations.flatten())
                rollout_truncated.copy_(batch.truncations.flatten())
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

            with timer.span("targets"), torch.no_grad():
                taken_logprob, next_q, next_entropy = target_inputs()
                temperature = actor.log_temperature.exp()
                targets = distribution_gve(
                    rollout_rewards.view(time_shape), (next_q + temperature * next_entropy).view(time_shape),
                    taken_logprob.view(time_shape), rollout_continuing.view(time_shape),
                    torch.maximum(rollout_truncated.view(time_shape), rollout_ends), temperature,
                    args.gamma, args.gve_lambda)
                rollout_targets.copy_(targets.flatten())

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
