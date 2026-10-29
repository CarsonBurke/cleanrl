# REPPO v52: v51 (v36 fast path, H8 latent roll, contrast head D) whose critic fits a replay window of the
# last W rollouts with off-policy-corrected soft trace targets.
# - Measured on v51: D's contrasts fit training states near the SNR ceiling (train corr 0.25) but transfer
#   poorly to fresh states (held-out corr 0.08), and one iteration of fitting the fresh 16k rows rewrites them
#   (drift ~ contrast std). Hypothesis: the contrast is variance-limited per state, so the lever is more
#   distinct states per fit, not more updates on the same ones. W rollouts give W x the states.
# - Targets (v11's V-trace with the sampled-action baseline, not v10's taken-action Retrace, which collapsed):
#     G_t = r_t + gamma V'_{t+1} + gamma c_{t+1} (G_{t+1} - E_pi Q'(s_{t+1}, .)),  c = lambda min(1, pi/mu)
#   V' keeps v51's soft value (K-sample Q' plus -T log pi bonus). The fresh rollout is treated as on-policy
#   (c = lambda), so W=1 reproduces v51's soft lambda-returns. Targets, D's targets and the latent-roll
#   features are recomputed over the whole window at every refresh; stored rows use the current obs norm.
# - Only the critic replays; the actor still trains on the fresh rollout, diagnostics stay on fresh rows.
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
CRITIC_METRICS = ("critic_ce", "aux_loss", "reward_loss", "q_mse", "support_clip_frac", "target_mean",
                  "anchor_loss", "anchor_rms", "contrast_ce", "contrast_scale", "contrast_train_corr",
                  "contrast_train_std_ratio")
ACTOR_METRICS = ("actor_objective", "q_pi", "entropy", "kl", "kl_violation", "temperature", "kl_multiplier",
                 "std", "saturation", "mean_abs_action")
DIAGNOSTIC_METRICS = ("contrast_corr_heldout", "contrast_std_ratio_heldout", "dqda_cos_consecutive", "dqda_norm", "advantage_corr_heldout", "advantage_std_ratio",
                      "contrast_drift_rms", "contrast_std")
# Per-refresh trace statistics, followed by one mean trace per slot age (0 = fresh).
TRACE_METRICS = ("mean_c", "ratio_clip_frac", "fresh_abs_log_ratio")


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
    """trace lambda of the critic target: c = lambda min(1, pi / mu) (lambda-returns on fresh data)"""
    replay_rollouts: int = 4
    """critic replay window W in rollouts; 1 = the fresh rollout only (v51)"""
    target_refresh: int = 16
    """critic updates between recomputations of the targets from the live critic"""
    bootstrap_samples: int = 4
    """current-policy actions averaged into each bootstrap's soft value"""
    diagnostic_lambda: float = 0.95
    """lambda of the held-out diagnostic returns (fixed across --gve-lambda)"""
    critic_updates: int = 128
    """critic updates per rollout, each on a uniform minibatch of the replay window"""
    actor_updates: int = 32
    """actor updates per rollout on fresh-rollout rows, interleaved evenly with the critic's"""
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
    temperature_source: str = "dual"
    """drift: T = drift_scale * rms contrast drift per iteration; dual: v36 entropy-target dual"""
    drift_scale: float = 1.0
    """multiplier on the rms contrast drift (sqrt of the critic's effective memory, in iterations)"""
    contrast_anchor: float = 0.0
    """kappa: weight of the previous iteration's contrasts, in samples per state (0 disables)"""
    anchor_samples: int = 8
    """policy actions per state at which contrasts are anchored"""
    actor_signal: str = "contrast"
    """contrast: climb s_y E[D(s,a)]; q: climb E[Z(s,a)] (v36)"""
    contrast_bins: int = 121
    """categories of the contrast head D"""
    contrast_range: float = 6.0
    """D's support is [-range, range] in units of s_y"""
    contrast_loss_coef: float = 1.0
    """weight of D's cross-entropy"""
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
        # D(s,a): categorical over the standardized contrast (G - E_pi Q') / s_y.
        self.contrast_head = nn.Sequential(nn.SiLU(), *normed_layer(width, width), nn.Linear(width, args.contrast_bins))
        self.contrast_hl = HLGaussConfig(v_min=-args.contrast_range, v_max=args.contrast_range,
                                         num_bins=args.contrast_bins).build(device)

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

    def contrast(self, observations, actions):
        """E[D(s,a)] in units of s_y."""
        return self.contrast_hl.to_scalar(self.contrast_head(self.features(observations, actions)))


class HostSquashedGaussian:
    """Reusable FP32 buffers for tanh-Gaussian sampling from host actor outputs.

    Besides the action, keeps the pre-tanh sample and its Gaussian log-density: the behavior density
    the replay's importance ratios are taken against (the tanh Jacobian cancels in the ratio).
    """

    def __init__(self, num_envs, action_dim):
        shape = (num_envs, action_dim)
        self.std = np.empty(shape, dtype=np.float32)
        self.noise = np.empty(shape, dtype=np.float32)
        self.pre = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)
        self.logprob = np.empty(num_envs, dtype=np.float32)
        self.offset = np.float32(action_dim * LOG_SQRT_2PI)

    def __call__(self, outputs, rng):
        mean, log_std = np.split(outputs, 2, axis=-1)
        np.exp(log_std, out=self.std)
        rng.standard_normal(self.noise.shape, dtype=np.float32, out=self.noise)
        np.multiply(self.noise, self.std, out=self.pre)
        self.pre += mean
        np.tanh(self.pre, out=self.action)
        np.einsum("ij,ij->i", self.noise, self.noise, out=self.logprob)
        self.logprob *= -0.5
        self.logprob -= log_std.sum(-1)
        self.logprob -= self.offset
        return self.action


def gaussian_logprob(pre, mean, std):
    return (-0.5 * ((pre - mean) / std).square() - std.log() - LOG_SQRT_2PI).sum(-1)


def shift_up(tensor):
    """Row t of the result is row t + 1 of a time-major tensor (zero past the end)."""
    return torch.cat((tensor[1:], torch.zeros_like(tensor[:1])))


def trace_returns(rewards, soft_values, sampled_q, traces, continuing, cuts, gamma):
    """Soft V-trace targets over time-major (T, columns) tensors, every column one environment of one slot.

    G_t = r_t + gamma cont_t [V'_t + (1 - cut_t) c_{t+1} (G_{t+1} - Qbar'_t)], with V'_t the soft value of
    s_{t+1} (K-sample Qbar'_t plus the expected entropy bonus), c_t the truncated trace of the stored action
    a_t, and cuts (truncations, rollout ends) bootstrapping from V'_t alone. With c = lambda this is v51's
    G_t = r_t + gamma [(1 - lambda) V'_t + lambda (G_{t+1} + T H^_{t+1})]. As y_t = x_t + a_t y_{t+1}:
    a_t = gamma cont_t (1 - cut_t) c_{t+1},  x_t = r_t + gamma cont_t V'_t - a_t Qbar'_t.
    """
    coefficients = gamma * continuing * (1.0 - cuts) * shift_up(traces)
    inputs = rewards + gamma * continuing * soft_values - coefficients * sampled_q
    return reverse_linear_scan(inputs, coefficients)


def validate_args(args):
    if args.actor_signal not in ("contrast", "q"):
        raise ValueError("--actor-signal must be contrast or q")
    if args.temperature_source not in ("drift", "dual"):
        raise ValueError("--temperature-source must be drift or dual")
    if min(args.num_envs, args.num_steps, args.minibatch_size, args.critic_updates, args.actor_updates,
           args.replay_rollouts) <= 0:
        raise ValueError("environment, rollout, minibatch, update and replay counts must be positive")
    if not 0.0 <= args.gve_lambda <= 1.0 or not 0.0 <= args.diagnostic_lambda <= 1.0:
        raise ValueError("trace lambdas must lie in [0, 1]")
    if args.aux_horizon <= 0 or args.aux_horizon >= args.num_steps:
        raise ValueError("aux_horizon must be in [1, num_steps)")
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
        actor_parameters = actor_network + dual_parameters
        target_entropy = -args.ent_target_mult * action_dim
        B, M, N = args.batch_size, args.minibatch_size, args.num_envs
        W, T = args.replay_rollouts, args.num_steps
        C = W * B

        # GPU ring of W rollout slots, preallocated at full size. Slot s holds rows [s B, (s + 1) B) in its
        # rollout's (step, env) order, so a row's environment is row % num_envs and its successor t + k is
        # row + k N (inside the slot whenever the chain mask says it is valid). Slots fill in order 0..W-1,
        # so the filled rows are always the prefix [0, filled_rows).
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
        # Per refresh, under the current actor and target critic: soft value V'(s'), its entropy bonus, and
        # log pi(u|s) - log mu(u|s) of the stored pre-tanh action u (zero on the fresh slot).
        replay_soft_values = torch.zeros(C, **f32)
        replay_bonus = torch.zeros(C, **f32)
        replay_log_ratio = torch.zeros(C, **f32)
        fresh_log_ratio = torch.zeros(B, **f32)
        # f'(s, a) of every replay row under the current target critic (recomputed at each refresh).
        target_features = torch.zeros((C, args.critic_hidden_dim), **f32)
        # replay_chain[k - 1, row]: row's episode continues k steps within its slot, so row + k N holds
        # s_{t+k}, a_{t+k}, r_{t+k}. Written once per slot at insertion.
        H = args.aux_horizon
        replay_chain = torch.zeros((H, C), **f32)
        diagnostic_targets = torch.zeros(C, **f32)
        slot_time_shape = (T, N)
        slot_ends = torch.zeros(slot_time_shape, **f32)
        slot_ends[-1] = 1.0
        # Traces run time-major with every (slot, env) pair its own column: (T, W N), cut at each slot's end.
        window_ends = slot_ends.repeat(1, W)
        window_rows = torch.arange(C, device=device)
        slot_rows = [window_rows[s * B:(s + 1) * B].clone() for s in range(W)]  # own storage: offset-free inputs
        filled_rows = torch.zeros((), **f32)
        fresh_start = torch.zeros((), dtype=torch.long, device=device)
        obs_mean = torch.zeros((N, observation_dim), **f32)
        obs_inv_std = torch.ones((N, observation_dim), **f32)
        # The normalization each drift critic was fit under (final: iteration k-1's, previous: k-2's).
        final_norm = torch.stack((obs_mean, obs_inv_std)).clone()
        previous_norm = final_norm.clone()
        # Anchor points: K policy actions per replay row and the previous critic's contrasts there.
        anchor_actions = torch.zeros((args.anchor_samples, C, action_dim), **f32)
        anchor_contrasts = torch.zeros((args.anchor_samples, C), **f32)
        anchor_weight = torch.zeros((), **f32)
        # Standardized contrast targets y = (G - E_pi Q'(s,.)) / s_y, refreshed with the trace targets.
        contrast_advantage = torch.zeros(C, **f32)
        contrast_targets = torch.zeros(C, **f32)
        contrast_scale = torch.ones((), **f32)
        critic_sums = torch.zeros(len(CRITIC_METRICS), **f32)
        actor_sums = torch.zeros(len(ACTOR_METRICS), **f32)
        grad_sums = torch.zeros(2, **f32)
        trace_sums = torch.zeros(len(TRACE_METRICS) + W, **f32)
        diagnostic_generator = torch.Generator(device=device).manual_seed(args.seed)

        def time_major(values):
            """(C,) slot-major rows -> (T, W N): column s N + e is environment e of slot s."""
            return values.view(W, T, N).transpose(0, 1).reshape(T, W * N)

        def slot_major(values):
            """Inverse of time_major."""
            return values.view(T, W, N).transpose(0, 1).reshape(C)

        def normalize(raw, rows):
            env = rows % N
            return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-args.obs_clip, args.obs_clip)

        def observations(rows):
            """Stored raw observations under the current normalization (every slot, every refresh)."""
            return normalize(replay_obs[rows], rows)

        def next_observations(rows):
            return normalize(replay_next_obs[rows], rows)

        def trace_inputs(rows):
            """Expected soft value V'(s') = mean_k [Q_frozen(s', a'_k) - T log pi(a'_k|s')] over K current-policy
            actions, the entropy bonus T H^(s') = -T mean_k log pi(a'_k|s'), and log pi(u|s) - log mu(u|s) of
            the stored pre-tanh action u under the current actor."""
            next_obs = next_observations(rows)
            mean, std = actor(next_obs)
            samples = args.bootstrap_samples
            next_actions, next_logprob = squashed_sample(mean.expand(samples, -1, -1), std.expand(samples, -1, -1))
            q = target_critic.value(next_obs.expand(samples, -1, -1), next_actions)
            bonus = -actor.log_temperature.exp() * next_logprob.mean(0)
            log_ratio = gaussian_logprob(replay_pre_actions[rows], *actor(observations(rows))) \
                - replay_behavior_logprob[rows]
            return q.mean(0) + bonus, bonus, log_ratio

        def critic_loss(rows):
            obs, actions, targets = observations(rows), replay_actions[rows], replay_targets[rows]
            # Successor rows t + k (k = 1..H), clamped in range; rows past their slot's end are masked by
            # the chain (it is zero from the slot's last step on).
            later = torch.minimum(rows + N * torch.arange(1, H + 1, device=device)[:, None], window_rows[-1])
            chain = replay_chain[:, rows]
            later_actions = replay_actions[later]
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
            reward_error = (critic.reward(latents[:-1]) - replay_rewards[reward_rows]).square()
            reward_loss = (reward_mask * reward_error).sum() / reward_mask.sum()
            q_mse = (critic.hl.to_scalar(logits.detach()) - targets).square().mean()
            with torch.no_grad():
                probs = logits.softmax(-1)
                support = critic.hl.support
                mean_z = (probs * support).sum(-1, keepdim=True)
                var_z = (probs * (support - mean_z).square()).sum(-1).mean()
            if args.contrast_anchor > 0.0:
                q_anchor = critic.value(obs.expand(args.anchor_samples, -1, -1), anchor_actions[:, rows])
                drift = (q_anchor - q_anchor.mean(0)) - anchor_contrasts[:, rows]
                anchor_msd = drift.square().mean()
                anchor = anchor_weight * anchor_msd / (2.0 * var_z)
            else:
                anchor_msd = anchor = torch.zeros((), device=device)
            contrast_logits = critic.contrast_head(features)
            contrast_labels = critic.contrast_hl.project(contrast_targets[rows])
            contrast_ce = -(contrast_labels * contrast_logits.log_softmax(dim=-1)).sum(-1).mean()
            with torch.no_grad():
                # On training rows: does D fit the targets (std ratio ~1) or stay compressed like Z?
                fitted = critic.contrast_hl.to_scalar(contrast_logits)
                fc, yc = fitted - fitted.mean(), contrast_targets[rows] - contrast_targets[rows].mean()
                train_corr = (fc * yc).mean() / (fc.std() * yc.std() + 1e-12)
                train_ratio = fc.std() / (yc.std() + 1e-12)
            support_clip = ((targets <= args.v_min) | (targets >= args.v_max)).float().mean()
            return ce + anchor + args.contrast_loss_coef * contrast_ce + args.aux_loss_coef * aux + args.reward_loss_coef * reward_loss, torch.stack((
                ce.detach(), aux.detach(), reward_loss.detach(), q_mse, support_clip, targets.mean(),
                anchor.detach(), anchor_msd.detach().sqrt(), contrast_ce.detach(), contrast_scale.clone(),
                train_corr, train_ratio))

        def actor_loss(rows):
            obs = observations(rows)
            with torch.no_grad():
                old_mean, old_std = policy_moments(old_actor, obs)
            mean, std = actor(obs)
            actions, logprob = squashed_sample(mean, std)
            if args.actor_signal == "contrast":
                q = contrast_scale * critic.contrast(obs, actions)
            else:
                q = critic.value(obs, actions)
            temperature = actor.log_temperature.exp()
            kl_multiplier = actor.log_kl_multiplier.exp()
            objective = -q + temperature.detach() * logprob
            kl = gaussian_kl(old_mean, old_std, mean, std)
            inside = kl < args.kl_bound
            # Outside the trust region a state only receives the pull back toward the old policy.
            policy_loss = torch.where(inside, objective, kl * kl_multiplier.detach()).mean()
            entropy = -logprob
            temperature_loss = (temperature * (entropy - target_entropy).detach().mean()
                                if args.temperature_source == "dual" else torch.zeros_like(temperature))
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
            critic_advantage = target_critic.value(obs, replay_actions[rows]) - baseline
            return_advantage = diagnostic_targets[rows] - baseline
            centered_c = critic_advantage - critic_advantage.mean()
            centered_r = return_advantage - return_advantage.mean()
            correlation = (centered_c * centered_r).mean() / (centered_c.std() * centered_r.std() + 1e-12)
            # Contrast drift: how much one iteration of fitting (previous_critic -> final_critic) moved the
            # action contrasts of policy samples, on states neither has been trained on.
            env = rows % N
            newer_obs = ((replay_obs[rows] - final_norm[0, env]) * final_norm[1, env]).clamp(-args.obs_clip, args.obs_clip)
            older_obs = ((replay_obs[rows] - previous_norm[0, env]) * previous_norm[1, env]).clamp(-args.obs_clip, args.obs_clip)
            newer_q = final_critic.value(newer_obs.expand(samples, -1, -1), sampled)
            older_q = previous_critic.value(older_obs.expand(samples, -1, -1), sampled)
            newer_a, older_a = newer_q - newer_q.mean(0), older_q - older_q.mean(0)
            drift = (newer_a - older_a).square().mean().sqrt()
            d_advantage = target_critic.contrast(obs, replay_actions[rows]) - target_critic.contrast(
                obs.expand(samples, -1, -1), sampled).mean(0)
            dc = d_advantage - d_advantage.mean()
            d_correlation = (dc * centered_r).mean() / (dc.std() * centered_r.std() + 1e-12)
            return torch.stack((d_correlation, dc.std() * contrast_scale / (centered_r.std() + 1e-12),
                                cosine, current.norm(dim=-1).mean(), correlation,
                                centered_c.std() / (centered_r.std() + 1e-12), drift, newer_a.std()))

        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16)

        def encode_targets(rows):
            """f'(s, a) of one slot's rows under the (just refreshed) target critic, and G - E_pi Q'(s, .)."""
            obs = observations(rows)
            features = target_critic.features(obs, replay_actions[rows])
            mean, std = actor(obs)
            samples = args.bootstrap_samples
            actions, _ = squashed_sample(mean.expand(samples, -1, -1), std.expand(samples, -1, -1))
            baseline = target_critic.value(obs.expand(samples, -1, -1), actions).mean(0)
            return features, replay_targets[rows] - baseline

        if args.compile:
            # Every call sees one slot (B rows), so each function compiles once for any W.
            encode_targets = graph_compile(encode_targets)
            trace_inputs = graph_compile(trace_inputs)
            critic_loss = graph_compile(critic_loss)
            actor_loss = graph_compile(actor_loss)

        def evaluate_slots(slots, fresh_slot):
            """Trace inputs of the given slots under the current actor and target critic. The fresh slot is
            on-policy by construction (ratio 1, as v51 treats it); its measured log-ratio is kept for logging."""
            for slot in slots:
                rows = slice(slot * B, (slot + 1) * B)
                soft_values, bonus, log_ratio = trace_inputs(slot_rows[slot])
                replay_soft_values[rows].copy_(soft_values)
                replay_bonus[rows].copy_(bonus)
                if slot == fresh_slot:
                    fresh_log_ratio.copy_(log_ratio)
                    replay_log_ratio[rows].zero_()
                else:
                    replay_log_ratio[rows].copy_(log_ratio)

        def window_targets(lam):
            """Soft trace targets of every window row from the last evaluate_slots (unevaluated or unfilled
            slots give stale values in their own columns only: columns never mix slots)."""
            traces = lam * replay_log_ratio.clamp(max=0.0).exp()
            returns = trace_returns(
                time_major(replay_rewards), time_major(replay_soft_values),
                time_major(replay_soft_values - replay_bonus), time_major(traces),
                time_major(replay_continuing), torch.maximum(time_major(replay_truncated), window_ends), args.gamma)
            return slot_major(returns), traces

        def uniform_rows(count):
            """Uniform rows in [0, count) from the default CUDA generator (graph-safe; count is a device scalar)."""
            return torch.minimum((torch.rand(M, device=device) * count).long(), count.long() - 1)

        def critic_step():
            critic_optimizer.zero_grad(set_to_none=False)
            with autocast():
                # Uniform over every filled slot of the window.
                loss, metrics = critic_loss(uniform_rows(filled_rows))
            loss.backward()
            grad_sums[1] += nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            critic_optimizer.step()
            critic_sums.add_(metrics)

        def actor_step():
            actor_optimizer.zero_grad(set_to_none=False)
            with autocast():
                # Fresh-rollout rows only: the actor's data is unchanged from v51.
                loss, metrics = actor_loss(fresh_start + torch.minimum(
                    (torch.rand(M, device=device) * B).long(), window_rows[B - 1]))
            # The critic is only differentiated through its action input.
            loss.backward(inputs=actor_parameters)
            grad_sums[0] += nn.utils.clip_grad_norm_(actor_network, args.max_grad_norm, foreach=True)
            actor_optimizer.step()
            actor_sums.add_(metrics)

        update_steps = None  # captured lazily, once the first rollout populated the buffers
        critic_per_actor = args.critic_updates // args.actor_updates
        refreshes = args.critic_updates // args.target_refresh

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

            # The fresh rollout overwrites the oldest slot; slots fill in order, so filled ones are a prefix.
            fresh_slot = (iteration - 1) % W
            fresh_offset = fresh_slot * B
            fresh = slice(fresh_offset, fresh_offset + B)
            filled = min(iteration, W)
            # Slot of each age (0 = fresh, filled - 1 = oldest), for the per-age trace metrics.
            age_slots = torch.tensor([(fresh_slot - age) % W for age in range(filled)], device=device)
            with timer.span("insert"), torch.no_grad():
                batch = transfer.upload()
                replay_obs[fresh].copy_(batch.fields["raw_obs"].flatten(0, 1))
                replay_next_obs[fresh].copy_(batch.fields["raw_next_obs"].flatten(0, 1))
                replay_actions[fresh].copy_(batch.fields["actions"].flatten(0, 1))
                replay_pre_actions[fresh].copy_(batch.fields["pre_actions"].flatten(0, 1))
                replay_behavior_logprob[fresh].copy_(batch.fields["behavior_logprob"].flatten())
                replay_rewards[fresh].copy_(batch.rewards.flatten())
                replay_continuing[fresh].copy_(1.0 - batch.terminations.flatten())
                replay_truncated[fresh].copy_(batch.truncations.flatten())
                filled_rows.fill_(float(filled * B))
                fresh_start.fill_(fresh_offset)
                previous_norm.copy_(final_norm)
                final_norm.copy_(torch.stack((obs_mean, obs_inv_std)))
                obs_mean.copy_(torch.from_numpy(obs_norm.means.astype(np.float32)))
                obs_inv_std.copy_(torch.from_numpy(
                    (1.0 / np.sqrt(obs_norm.variances + obs_norm.epsilon)).astype(np.float32)))
                torch._foreach_copy_(list(previous_critic.parameters()), list(final_critic.parameters()))
                torch._foreach_copy_(list(final_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                torch._foreach_copy_(list(old_actor.parameters()), actor_network)

            with torch.no_grad():
                # The episode survives step t -> t + 1 inside the slot unless it terminated, truncated, or the
                # rollout ended at t; a slot's chains depend only on its own rollout, so only the fresh one is new.
                cuts = torch.maximum(replay_truncated[fresh].view(slot_time_shape), slot_ends)
                survives = replay_continuing[fresh].view(slot_time_shape) * (1.0 - cuts)
                chain = torch.ones_like(survives)
                for k in range(H):
                    chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
                    replay_chain[k, fresh].copy_(chain.flatten())

            def refresh_window():
                """Trace targets, latent-roll features and D's targets of every filled slot under the current
                actor and (just refreshed) target critic; accumulates the trace metrics."""
                evaluate_slots(range(filled), fresh_slot)
                targets, traces = window_targets(args.gve_lambda)
                replay_targets.copy_(targets)
                for slot in range(filled):
                    rows = slice(slot * B, (slot + 1) * B)
                    features, advantage = encode_targets(slot_rows[slot])
                    target_features[rows].copy_(features)
                    contrast_advantage[rows].copy_(advantage)
                live = slice(0, filled * B)
                # One scale over the window (D's head is shared by every slot); unfilled rows stay zero.
                contrast_scale.copy_(contrast_advantage[live].std().clamp_min(1e-3))
                torch.div(contrast_advantage, contrast_scale, out=contrast_targets)
                slot_traces = traces.view(W, B).mean(1)
                # Fraction of replayed (non-fresh) rows whose ratio pi/mu > 1 is clipped to 1 (the fresh
                # slot's log-ratio is zeroed, so it never counts).
                clipped = (replay_log_ratio[live] > 0.0).float().sum() / max((filled - 1) * B, 1)
                trace_sums[:len(TRACE_METRICS)] += torch.stack((
                    slot_traces[:filled].mean(), clipped, fresh_log_ratio.abs().mean()))
                trace_sums[len(TRACE_METRICS):len(TRACE_METRICS) + filled] += slot_traces[age_slots]

            with timer.span("targets"), torch.no_grad():
                # Held-out diagnostic returns of the fresh rollout (on-policy: c = diagnostic_lambda). Columns
                # never mix slots, so the other slots' stale inputs do not reach the fresh slot's returns.
                evaluate_slots((fresh_slot,), fresh_slot)
                diagnostic_targets[fresh].copy_(window_targets(args.diagnostic_lambda)[0][fresh])

            with timer.span("diagnostics"), torch.no_grad():
                # The actor still equals the rollout's behavior policy (no update yet this iteration).
                diagnostic_rows = fresh_offset + torch.randperm(
                    B, device=device, generator=diagnostic_generator)[: args.diagnostic_rows]
                diagnostics = critic_diagnostics(diagnostic_rows)
                if args.contrast_anchor > 0.0 and iteration > 1:
                    # Iteration 1's predecessor is the random initialization: nothing to remember.
                    anchor_weight.fill_(args.contrast_anchor)
                    # Every filled row is a critic training row, so every one gets anchor points.
                    for start in range(0, filled * B, args.diagnostic_rows):
                        rows = window_rows[start:start + args.diagnostic_rows]
                        env = rows % N
                        mean, std = actor(observations(rows))
                        samples = mean.expand(args.anchor_samples, -1, -1)
                        noise = torch.randn(samples.shape, device=device, generator=diagnostic_generator)
                        actions = torch.tanh(samples + std * noise)
                        prev_obs = ((replay_obs[rows] - final_norm[0, env]) * final_norm[1, env]).clamp(
                            -args.obs_clip, args.obs_clip)
                        q_prev = final_critic.value(prev_obs.expand(args.anchor_samples, -1, -1), actions)
                        anchor_actions[:, rows] = actions
                        anchor_contrasts[:, rows] = q_prev - q_prev.mean(0)
                if args.temperature_source == "drift" and iteration > 2:
                    # Iterations 1-2 have no fitted predecessor pair; they keep ent_start.
                    actor.log_temperature.copy_((args.drift_scale * diagnostics[6]).clamp_min(1e-8).log())

            profiler = None
            if iteration == args.profile_iteration:
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                profiler.__enter__()
            # "update" excludes the window refreshes, timed as "refresh" (v51's update_s = update + refresh).
            timer.start("update")
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
            trace_sums.zero_()
            for update in range(1, args.critic_updates + 1):
                if (update - 1) % args.target_refresh == 0:
                    timer.stop()
                    timer.start("refresh")
                    with torch.no_grad():
                        if update > 1:
                            torch._foreach_copy_(list(target_critic.parameters()), critic_parameters)
                        refresh_window()
                    timer.stop()
                    timer.start("update")
                critic_update()
                if update % critic_per_actor == 0:
                    actor_update()
            timer.stop()

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
                **{f"trace/{name}": trace_sums[i] / refreshes for i, name in enumerate(TRACE_METRICS)},
                **{f"trace/mean_c_age{age}": trace_sums[len(TRACE_METRICS) + age] / refreshes
                   for age in range(filled)},
                "trace/mean_c_oldest": trace_sums[len(TRACE_METRICS) + filled - 1] / refreshes,
                "trace/filled_slots": filled_rows / B,
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
