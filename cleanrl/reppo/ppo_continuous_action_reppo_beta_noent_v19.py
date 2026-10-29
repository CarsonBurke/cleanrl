# REPPO v19: v18 (Beta policy, mode parameterization) with no entropy regularization at all.
# v18 and every tanh-Gaussian REPPO here share a return dip at ~1.2-1.6M: the entropy temperature winds
# down to ~0.001 while entropy is above target, lags when it crosses, then overshoots to 0.05-0.09; the
# T H / (1 - gamma) term it adds to every soft return shifts all Q targets by ~20-35% within a few
# iterations. The Beta head keeps its own uncertainty: alpha, beta >= 1 bound it away from U-shapes, its
# spread is a learned output, and each update moves it only within the KL trust region. Removed: the
# temperature and its dual, the entropy target, and the soft terms of the returns and the actor
# objective. The critic learns the plain Q^pi(s, a) with lambda-returns; the actor maximizes
# E_{a~pi} Q(s, a) under KL(old || new) <= kl_bound. Entropy is still logged.
# Hypothesis: without the dual's windup and the soft-return shock the dip disappears, and the Beta's
# pathwise, trust-region-limited sharpening keeps enough exploration on its own.
#
# v18 notes:
# REPPO v18: v7 at 128 critic updates (cu128, 10.5k) with a Beta policy in place of the tanh-Gaussian.
# Physical action a = 2x - 1, x ~ Beta(alpha, beta), alpha, beta >= 1 (unimodal, bounded, no squashing).
# - No tanh: the pathwise gradient dQ/da is not multiplied by 1 - tanh^2(u), the tanh-Gaussian's dead
#   zone near the bounds where HalfCheetah's bang-bang optimum lives (Beta's own dx/dtheta still shrinks
#   as the mode nears a bound, but smoothly). It is Beta's implicit reparameterization (Dirichlet rsample).
# - Every entropy and KL is closed-form: the actor maximizes Q(s, a) + T H(pi(.|s)), the soft reward adds
#   gamma T H(pi(.|s')) (Q'(s', a') still at a sampled a'), and the trust region is the exact
#   KL(old || new). The tanh-Gaussian used one-sample -log pi for all entropy terms.
# - Parameterization (--parameterization):
#   mode       alpha = 1 + m nu, beta = 1 + (1 - m) nu, m = sigmoid(z_m) the mode, nu = exp(z_nu): the
#              Beta analogue of (mean, log std). Mode sensitivity dm/dz_m does not shrink as the policy
#              sharpens, and precision moves multiplicatively.
#   softplus   alpha, beta = 1 + softplus(z) (the PPO Beta head; reppo_ngpt_beta_v3). The mean moves by
#              ~1/(alpha + beta) per unit logit, so the mean's gradient vanishes as the policy sharpens,
#              and precision grows only linearly in the logits.
# The earlier Beta REPPO (v3) is no evidence against Beta: it ran on the weaker ngpt base (v2: 4.6k)
# at lr 2.4e-3 and KL bound 0.1. Entropies are physical (native + log 2 per dimension), so the target
# -0.5 x action_dim means what it did for the tanh-Gaussian.
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
from torch.distributions import Beta
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
ZERO_DIST_SCALE = 40.9  # REPPO's fixed multiplier on the learned zero-value logit offset
CRITIC_METRICS = ("critic_ce", "aux_loss", "q_mse", "support_clip_frac", "target_mean")
ACTOR_METRICS = ("actor_objective", "q_pi", "entropy", "kl", "kl_violation", "kl_multiplier", "std",
                 "concentration")


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
    """lambda of the lambda-return critic target"""
    critic_updates: int = 128
    """critic updates per rollout, each on a uniform replay minibatch (v4: 4 epochs x 8)"""
    actor_updates: int = 32
    """actor updates per rollout, interleaved evenly with the critic's"""
    buffer_iterations: int = 1
    """replay capacity in rollouts (61 x 16384 ~ 1M transitions); 1 = the fresh rollout only"""
    actor_replay: bool = False
    """sample actor states from the whole replay instead of the fresh rollout"""
    coupled_minibatch: bool = False
    """the actor updates on the rows of the critic update just before it (v4)"""
    epoch_sampling: bool = False
    """critic rows are epoch permutations of the fresh rollout (v4), not uniform draws"""
    rollout_normalization: bool = False
    """train on observations as normalized during the rollout (v4), not re-normalized"""
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
    parameterization: str = "mode"
    """Beta head: "mode" (sigmoid mode, exp concentration) or "softplus" (alpha, beta = 1 + softplus)"""
    native_epsilon: float = 1e-6
    """native samples are clipped to [eps, 1 - eps] before the action map"""
    kl_bound: float = 0.03
    """per-state KL(old || new) trust region"""
    kl_start: float = 0.01
    """initial KL Lagrange multiplier"""
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


def beta_actions(alpha, beta, epsilon, reparameterized):
    policy = Beta(alpha, beta, validate_args=False)
    native = policy.rsample() if reparameterized else policy.sample()
    return 2.0 * native.clamp(epsilon, 1.0 - epsilon) - 1.0


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, args):
        super().__init__()
        width = args.actor_hidden_dim
        # Flat stack so the fused host mirror can evaluate it in one native call.
        self.net = nn.Sequential(*normed_layer(observation_dim, width), *normed_layer(width, width),
                                 nn.Linear(width, 2 * action_dim))
        self.parameterization = args.parameterization
        head = self.net[-1]
        with torch.no_grad():
            # Near-state-independent start at alpha = beta = 1 + softplus(0) ~ 1.69 (physical std ~ 0.48)
            # under either parameterization, not O(1) random concentrations.
            head.weight.mul_(0.01)
            head.bias.zero_()
            if self.parameterization == "mode":
                head.bias[action_dim:] = float(np.log(2.0 * np.log(2.0)))
        self.log_kl_multiplier = nn.Parameter(torch.tensor(float(np.log(args.kl_start))))

    def forward(self, observations):
        return beta_concentrations(self.net, observations, self.parameterization)


def beta_concentrations(net, observations, parameterization):
    first, second = net(observations).chunk(2, dim=-1)
    if parameterization == "softplus":
        return 1.0 + F.softplus(first), 1.0 + F.softplus(second)
    # sigmoid(-z) is 1 - mode without float32 cancellation near the bounds.
    precision = second.exp()
    return 1.0 + torch.sigmoid(first) * precision, 1.0 + torch.sigmoid(-first) * precision


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


class HostModeBeta:
    """Host Beta head for the mode parameterization: the float32 concentrations, then rng.beta."""

    def __init__(self, num_envs, action_dim, epsilon):
        shape = (num_envs, action_dim)
        self.epsilon = epsilon
        self.mode = np.empty(shape, dtype=np.float32)
        self.complement = np.empty(shape, dtype=np.float32)
        self.alpha = np.empty(shape, dtype=np.float32)
        self.beta = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, outputs, rng):
        first, second = np.split(outputs, 2, axis=-1)
        # mode = sigmoid(z) and 1 - mode = sigmoid(-z), each without cancellation.
        np.negative(first, out=self.mode)
        np.exp(self.mode, out=self.mode)
        np.exp(first, out=self.complement)
        self.mode += 1.0
        self.complement += 1.0
        np.reciprocal(self.mode, out=self.mode)
        np.reciprocal(self.complement, out=self.complement)
        np.exp(second, out=self.beta)
        np.multiply(self.mode, self.beta, out=self.alpha)
        self.beta *= self.complement
        self.alpha += 1.0
        self.beta += 1.0
        np.copyto(self.action, rng.beta(self.alpha, self.beta), casting="same_kind")
        np.clip(self.action, self.epsilon, 1.0 - self.epsilon, out=self.action)
        self.action *= 2.0
        self.action -= 1.0
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
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates,
           args.buffer_iterations) <= 0:
        raise ValueError("environment, rollout, minibatch, update and buffer counts must be positive")
    if args.parameterization not in ("mode", "softplus") or not 0.0 < args.native_epsilon < 0.5:
        raise ValueError("parameterization must be mode or softplus, native_epsilon in (0, 0.5)")
    if args.critic_updates % args.actor_updates:
        raise ValueError("actor_updates must divide critic_updates (even interleaving)")
    if args.coupled_minibatch and args.critic_updates != args.actor_updates:
        raise ValueError("coupled minibatches need one actor update per critic update")
    if (args.epoch_sampling or args.rollout_normalization) and args.buffer_iterations != 1:
        raise ValueError("epoch sampling and rollout normalization are defined for the fresh rollout only")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.buffer_size = args.buffer_iterations * args.batch_size
    if args.epoch_sampling and (args.batch_size % args.minibatch_size
                                or (args.critic_updates * args.minibatch_size) % args.batch_size):
        raise ValueError("epoch sampling needs whole epochs of whole minibatches")
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
            raise ValueError("the Beta policy's 2 x - 1 action map requires a [-1, 1] Box action space")
        obs_shape = envs.single_observation_space.shape
        observation_dim, action_dim = int(np.prod(obs_shape)), int(np.prod(space.shape))
        actor = Actor(observation_dim, action_dim, args).to(device)
        critic = Critic(observation_dim, action_dim, args, device).to(device)
        # Frozen per-iteration copies: the critic's bootstrap target and the KL anchor.
        target_critic = copy.deepcopy(critic).requires_grad_(False)
        old_actor = copy.deepcopy(actor.net).requires_grad_(False)
        actor_network, critic_parameters = list(actor.net.parameters()), list(critic.parameters())
        dual_parameters = [actor.log_kl_multiplier]
        actor_optimizer = torch.optim.AdamW(
            [{"params": actor_network},
             {"params": dual_parameters, "lr": args.dual_step_per_iteration / args.actor_updates,
              "weight_decay": 0.0}],
            lr=args.learning_rate, fused=True)
        critic_optimizer = torch.optim.AdamW(critic_parameters, lr=args.learning_rate, fused=True)
        actor_parameters = actor_network + dual_parameters
        B, M, C, N = args.batch_size, args.minibatch_size, args.buffer_size, args.num_envs

        # GPU replay of raw observations. Rows keep the rollout's (step, env) flattening and slots
        # are whole rollouts, so a row's environment is row % num_envs.
        f32 = dict(device=device, dtype=torch.float32)
        replay_obs = torch.zeros((C, observation_dim), **f32)
        replay_next_obs = torch.zeros((C, observation_dim), **f32)
        replay_actions = torch.zeros((C, action_dim), **f32)
        replay_rewards = torch.zeros(C, **f32)
        replay_continuing = torch.zeros(C, **f32)
        # v4's inputs: observations as the rollout normalized them (fresh rollout only).
        rollout_obs = torch.zeros((C, observation_dim), **f32) if args.rollout_normalization else None
        rollout_next_obs = torch.zeros((C, observation_dim), **f32) if args.rollout_normalization else None
        # Critic rows of the current update (read by a coupled actor) and the epoch schedule.
        critic_rows = torch.zeros(M, dtype=torch.long, device=device)
        schedule = torch.zeros((args.critic_updates, M), dtype=torch.long, device=device)
        cursor = torch.zeros(1, dtype=torch.long, device=device)
        replay_truncated = torch.zeros(C, **f32)
        replay_targets = torch.zeros(C, **f32)
        replay_next_q = torch.zeros(C, **f32)
        # Time-major (buffer_iterations * num_steps, num_envs) view of the ring; every stored rollout
        # ends in a bootstrap, so rollouts (and the ring's wrap point) never share a lambda-chain.
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
            if args.rollout_normalization:
                return rollout_obs[rows]
            return normalize(replay_obs[rows], rows)

        def next_observations(rows):
            if args.rollout_normalization:
                return rollout_next_obs[rows]
            return normalize(replay_next_obs[rows], rows)

        def bootstrap_values(rows):
            """Q_frozen(s', a') at a fresh current-policy action."""
            next_obs = next_observations(rows)
            next_actions = beta_actions(*actor(next_obs), args.native_epsilon, reparameterized=False)
            return target_critic.value(next_obs, next_actions)

        def critic_loss(rows):
            obs, next_obs = observations(rows), next_observations(rows)
            actions, continuing, targets = replay_actions[rows], replay_continuing[rows], replay_targets[rows]
            with torch.no_grad():
                next_actions = beta_actions(*actor(next_obs), args.native_epsilon, reparameterized=False)
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
                old_alpha, old_beta = beta_concentrations(old_actor, obs, args.parameterization)
            alpha, beta = actor(obs)
            q = critic.value(obs, beta_actions(alpha, beta, args.native_epsilon, reparameterized=True))
            kl_multiplier = actor.log_kl_multiplier.exp()
            objective = -q
            kl = beta_kl(old_alpha, old_beta, alpha, beta)
            inside = kl < args.kl_bound
            # Outside the trust region a state only receives the pull back toward the old policy.
            policy_loss = torch.where(inside, objective, kl * kl_multiplier.detach()).mean()
            multiplier_loss = -kl_multiplier * (kl.mean() - args.kl_bound).detach()
            metrics = torch.stack((objective.mean(), q.mean(), physical_entropy(alpha, beta).mean(), kl.mean(),
                                   1.0 - inside.float().mean(), kl_multiplier,
                                   beta_std(alpha, beta).mean(), (alpha + beta).mean())).detach()
            return policy_loss + multiplier_loss, metrics

        if args.compile:
            bootstrap_values = graph_compile(bootstrap_values)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        def uniform_rows(count):
            """Uniform replay rows from the default CUDA generator (graph-safe, fresh per replay)."""
            rows = (torch.rand(M, device=device) * count).long()
            return torch.minimum(rows, count.long() - 1)

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            if args.epoch_sampling:
                critic_rows.copy_(schedule.index_select(0, cursor).squeeze(0))
                cursor.add_(1)
            else:
                critic_rows.copy_(uniform_rows(filled))
            loss, metrics = critic_loss(critic_rows)
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            critic_sums.add_(metrics)

        fresh_count = torch.full((), float(B), **f32)

        def actor_step():
            if args.coupled_minibatch:
                rows = critic_rows
            elif args.actor_replay:
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
        # One Triton kernel; a cudagraph-tree wrapper would fault the manually captured update graphs.
        gae_fn = get_gae_fn(compiled=args.compile, mode="default", explicit_next_values=True)
        critic_per_actor = args.critic_updates // args.actor_updates

        host_actor = make_host_mirror(actor.net, args.num_envs)
        writer.add_text("host_actor", f"fused={host_actor.fused} fallback={host_actor.fallback_reason}")
        sampler = np.random.default_rng(args.seed)
        if args.parameterization == "softplus":
            beta_head = make_beta_sampler(args.num_envs, action_dim, space.low.reshape(-1).astype(np.float32),
                                          space.high.reshape(-1).astype(np.float32), epsilon=args.native_epsilon)
            writer.add_text("host_sampler", f"fused={beta_head.fused} fallback={beta_head.fallback_reason}")

            def sample_actions(outputs, rng):
                return beta_head(outputs, rng)[1]
        else:
            sample_actions = HostModeBeta(args.num_envs, action_dim, args.native_epsilon)

        def act(observations):
            action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return action

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"raw_obs": obs_shape, "raw_next_obs": obs_shape,
                                           "actions": (action_dim,), "obs": obs_shape, "next_obs": obs_shape})
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
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    np.copyto(raw_next, raw_obs)
                    for index in np.flatnonzero(np.logical_or(terms, truncs)):
                        raw_next[index] = infos["final_observation"][index]
                    transfer.push(step, np.asarray(raw_reward) * args.reward_scale, terms, truncs,
                                  raw_obs=raw_current, raw_next_obs=raw_next, actions=action,
                                  obs=obs_step, next_obs=transition_obs)
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
                replay_rewards[rows].copy_(batch.rewards.flatten())
                replay_continuing[rows].copy_(1.0 - batch.terminations.flatten())
                replay_truncated[rows].copy_(batch.truncations.flatten())
                if args.rollout_normalization:
                    rollout_obs[rows].copy_(batch.fields["obs"].flatten(0, 1))
                    rollout_next_obs[rows].copy_(batch.fields["next_obs"].flatten(0, 1))
                if args.epoch_sampling:
                    epochs = args.critic_updates * M // B
                    schedule.copy_(torch.cat([torch.randperm(B, device=device) for _ in range(epochs)]).view_as(schedule))
                filled.fill_(float(min(iteration, args.buffer_iterations) * B))
                fresh_start.fill_(slot)
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

            with timer.span("targets"), torch.no_grad():
                # Re-evaluate every stored rollout's GVE targets under the current policy and critic.
                for stored in range(min(iteration, args.buffer_iterations)):
                    rows = rollout_rows + stored * B
                    replay_next_q[stored * B : (stored + 1) * B].copy_(bootstrap_values(rows))
                targets = lambda_returns(
                    gae_fn, replay_rewards.view(time_shape), replay_next_q.view(time_shape),
                    (1.0 - replay_continuing).view(time_shape),
                    torch.maximum(replay_truncated.view(time_shape), rollout_ends),
                    args.gamma, args.gve_lambda)
                replay_targets.copy_(targets.flatten())

            with timer.span("update"):
                if update_steps is None:
                    if args.compile:
                        graphs = capture_steps([critic_step, actor_step],
                                               optimizers=[critic_optimizer, actor_optimizer],
                                               state_tensors=[critic_sums, actor_sums, grad_sums,
                                                              critic_rows, cursor])
                        update_steps = [graph.replay for graph in graphs]
                    else:
                        update_steps = [critic_step, actor_step]
                critic_update, actor_update = update_steps
                critic_sums.zero_()
                actor_sums.zero_()
                grad_sums.zero_()
                cursor.zero_()
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
