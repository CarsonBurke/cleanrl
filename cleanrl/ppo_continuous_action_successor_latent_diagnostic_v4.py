# Successor-latent diagnosis v4: fresh initialization, no checkpoint loading.
# Factorial decoder probes separate fitting/representation/context limitations.
# Longer fresh holdout audits reference signal and same-state horizon/tail sensitivity.
# The critic generates a joint future-transition latent at T ~ Geom(1-gamma).
# Its multi-step distributional Bellman loss contains NO rewards or returns.
# A separately fitted nonlinear immediate-reward decoder supplies latent costates.
# Coordinate-counterfactual latent predictions yield one advantage per action;
# no scalar GAE enters that estimator. Pathwise credit is a secondary diagnostic.
# Frozen-policy Monte Carlo score gradients audit model gradient bias before RL.
import copy
import json
import math
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import TruncationBootstrapCache, explained_variance, gather_metrics, get_gae_fn
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


@dataclass
class Args:
    exp_name: str = "successor_latent_diagnostic_v4_fresh"
    seed: int = 1
    env_id: str = "HalfCheetah-v4"
    num_envs: int = 16
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    calibration_rollouts: int = 32
    fit_rollouts: int = 128
    holdout_rollouts: int = 256
    decoder_width: int = 256
    bellman_steps: int = 32
    update_epochs: int = 10
    learning_rate: float = 3e-4
    value_learning_rate: float = 9.6e-3
    max_grad_norm: float = 0.5
    noise_dim: int = 32
    latent_dim: int = 64
    critic_width: int = 128
    eval_projections: int = 32
    eval_samples: int = 4
    gradient_block_steps: int = 256
    env_backend: str = "auto"
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    capture_video: bool = False
    torch_deterministic: bool = True
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    save_model: bool = True


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ValueCritic(nn.Module):
    def __init__(self, observation_dim, *, placement, norm_kind, activation):
        super().__init__()
        self.trunk = make_norm_residual_trunk(
            observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation
        )
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, observations):
        return self.value_head(self.trunk(observations))


class Agent(nn.Module):
    def __init__(self, envs, *, placement="pre", norm_kind="rms", activation="stiglu"):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = ValueCritic(observation_dim, placement=placement, norm_kind=norm_kind, activation=activation)
        self.actor = nn.Sequential(
            make_norm_residual_trunk(observation_dim, 64, placement=placement, norm_kind=norm_kind, activation=activation),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

    def get_value(self, observations):
        return self.critic(observations)

    def get_policy_and_value(self, observations):
        alpha, beta = (F.softplus(self.actor(observations)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(observations)

    def action_logprob(self, alpha, beta, native_action):
        return (Beta(alpha, beta, validate_args=False).log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, observations, action=None):
        alpha, beta, value = self.get_policy_and_value(observations)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((observations.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(observations.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        return action, self.action_logprob(alpha, beta, native), (distribution.entropy() + self.log_action_scale).sum(-1), value


def sample_parameter_directions(actor, count, generator):
    return {
        name: torch.empty((count,) + tuple(parameter.shape), device=parameter.device, dtype=parameter.dtype)
        .bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        for name, parameter in actor.named_parameters()
    }


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id,
        args.num_envs,
        backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video,
        run_name=run_name,
    )


class FrozenObsNorm:
    """Read-only checkpoint preprocessing for the gate, not a training normalizer.

    Shared VectorObsNorm remains the training implementation. Ping-pong outputs
    preserve the old observation until RolloutTransfer stages the same transition.
    """
    def __init__(self, state, num_envs, obs_shape):
        self.means = state["means"].cpu().numpy().copy()
        self.variances = state["variances"].cpu().numpy().copy()
        self.counts = state["counts"].cpu().numpy().copy()
        self.epsilon, self.clip = state["epsilon"], state["clip"]
        if self.means.shape != (num_envs, int(np.prod(obs_shape))):
            raise ValueError("gate must use the checkpoint's environment count and observation shape")
        self.inverse_std = 1.0 / np.sqrt(self.variances + self.epsilon)
        self._buffers = [np.empty((num_envs,) + tuple(obs_shape), dtype=np.float32) for _ in range(2)]
        self._transition = np.empty_like(self._buffers[0])
        self._work = np.empty_like(self.means, dtype=np.float64)
        self._cursor = 0

    def normalize(self, obs, rows=None, out_dtype=np.float32):
        if rows is not None:
            return np.clip((obs - self.means[rows]) * self.inverse_std[rows], -self.clip, self.clip).astype(out_dtype)
        output = self._buffers[self._cursor]
        self._cursor ^= 1
        np.subtract(np.asarray(obs).reshape(self.means.shape), self.means, out=self._work)
        np.multiply(self._work, self.inverse_std, out=self._work)
        np.clip(self._work, -self.clip, self.clip, out=self._work)
        output[...] = self._work.reshape(output.shape)
        return output

    def normalize_step(self, raw_next_obs, terminations, truncations, infos):
        normalized = self.normalize(raw_next_obs)
        self._transition[...] = normalized
        for index in np.flatnonzero(terminations | truncations):
            finals = infos.get("final_observation")
            if finals is None or finals[index] is None:
                raise ValueError("truncation requires its final observation, never the autoreset observation")
            final = np.asarray(finals[index]).reshape(-1)
            self._transition[index] = np.clip(
                (final - self.means[index]) * self.inverse_std[index], -self.clip, self.clip
            ).reshape(self._transition[index].shape)
        return normalized, self._transition


class FrozenRewardNorm:
    """Freeze a calibrated shared VectorRewardNorm's scaling for fit and holdout."""
    def __init__(self, normalizer):
        self.means, self.variances, self.counts = (getattr(normalizer, name).copy() for name in ("means", "variances", "counts"))
        self.epsilon, self.clip = normalizer.epsilon, normalizer.clip
        self.inverse_std = 1.0 / np.sqrt(self.variances + self.epsilon)
        self._work = np.empty_like(self.variances)
        self._output = np.empty_like(self.variances, dtype=np.float32)

    def normalize(self, rewards, terminations, out_dtype=np.float32):
        np.multiply(rewards, self.inverse_std, out=self._work)
        np.clip(self._work, -self.clip, self.clip, out=self._work)
        self._output[...] = self._work
        return self._output


def transition_features(observations, actions, next_observations):
    """Complete, fixed reconstruction targets for the learned transition encoder.

    State, action, and displacement each retain their supervised coordinates;
    reward does not select which coordinates the encoder must reconstruct.
    """
    # The final coordinate distinguishes a real zero-valued transition from the
    # absorbing all-zero outcome used beyond true termination.
    return torch.cat((observations, 2 * actions - 1, (next_observations - observations) / math.sqrt(2),
                      torch.ones_like(observations[:, :1])), -1)


class SuccessorCritic(nn.Module):
    """Sample a joint transition latent, NOT its conditional mean or a return."""
    def __init__(self, observation_dim, action_dim, noise_dim=32, width=128, latent_dim=None):
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim or 2 * observation_dim + action_dim
        self.network = nn.Sequential(
            layer_init(nn.Linear(observation_dim + action_dim + noise_dim, width)), nn.SiLU(),
            layer_init(nn.Linear(width, width)), nn.SiLU(),
            layer_init(nn.Linear(width, width)), nn.SiLU(),
            layer_init(nn.Linear(width, self.latent_dim), std=0.1),
        )

    def forward(self, observations, actions, noise):
        return self.network(torch.cat((observations, 2 * actions - 1, noise), -1))


class TransitionRepresentation(nn.Module):
    """Reward-independent learned latent, calibrated then frozen before Bellman fit.

    Reconstruction preserves the full transition; covariance regularization
    resists degenerate coordinates. Neither is a proof of invertibility, so real
    heldout reconstruction error remains part of the promotion gate.
    """
    def __init__(self, transition_dim, latent_dim=64, width=128):
        super().__init__()
        self.encoder = nn.Sequential(layer_init(nn.Linear(transition_dim, width)), nn.SiLU(),
                                     layer_init(nn.Linear(width, latent_dim), std=1.))
        self.decoder = nn.Sequential(layer_init(nn.Linear(latent_dim, width)), nn.SiLU(),
                                     layer_init(nn.Linear(width, transition_dim), std=1.))

    def forward(self, transitions):
        # Preserve the absorbing origin independently of learned biases.
        return self.encoder(transitions) - self.encoder(torch.zeros_like(transitions[:1]))

    def reconstruct(self, latents):
        return self.decoder(latents) - self.decoder(torch.zeros_like(latents[:1]))

    def loss(self, transitions):
        latent = self(transitions)
        reconstruction = (self.reconstruct(latent) - transitions).square().mean()
        centered = latent - latent.mean(0)
        covariance = centered.T @ centered / latent.shape[0]
        identity = torch.eye(latent.shape[-1], device=latent.device, dtype=latent.dtype)
        geometry = (covariance - identity).square().sum() / latent.shape[-1]
        return reconstruction + .1 * geometry


class RewardDecoder(nn.Module):
    def __init__(self, latent_dim, width=128):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(latent_dim, width)), nn.SiLU(),
            layer_init(nn.Linear(width, width)), nn.SiLU(),
            layer_init(nn.Linear(width, 1), std=0.1),
        )

    def forward(self, latent):
        # Zero is the absorbing outcome; its utility is identically zero.
        return (self.network(latent) - self.network(torch.zeros_like(latent[:1]))).squeeze(-1)


def energy_score(first, second, target):
    """Unbiased two-sample energy score; Euclidean, not squared distance.

    The repulsive term is essential: without it the optimum is a point forecast.
    Coordinates are scored jointly so their dependence remains identifiable.
    """
    scale = math.sqrt(first.shape[-1])
    return (0.5 * ((first - target).norm(dim=-1) + (second - target).norm(dim=-1))
            - 0.5 * (first - second).norm(dim=-1)) / scale


def trajectory_spans(terminations, truncations, horizon):
    """How many observed transitions remain before a reset or rollout boundary."""
    spans = np.empty(terminations.shape, dtype=np.int64)
    following = np.zeros(terminations.shape[1], dtype=np.int64)
    for step in range(len(spans) - 1, -1, -1):
        following = np.where(terminations[step] | truncations[step], 1, following + 1)
        spans[step] = np.minimum(following, horizon)
    return spans


def successor_indices(spans, geometric_horizons, num_envs):
    rows = torch.arange(spans.numel(), device=spans.device)
    observed = geometric_horizons < spans
    selected = rows + torch.minimum(geometric_horizons, spans - 1) * num_envs
    return observed, selected


def draw_geometric(shape, gamma, device):
    # T=0 denotes the current transition: P(T=t)=(1-gamma)*gamma**t.
    # 1-rand is in (0,1], avoiding an artificial finite-horizon cutoff.
    return (torch.log1p(-torch.rand(shape, device=device)) / math.log(gamma)).floor().long()


def select_successor_target(observed, selected, latents, terminations, bootstrapped):
    future = torch.where(terminations[selected, None], torch.zeros_like(bootstrapped), bootstrapped)
    return torch.where(observed[:, None], latents[selected], future)


def make_logit_projector(actor):
    parameters = dict(actor.named_parameters())

    def project(observations, directions):
        def forward(parameters):
            return torch.func.functional_call(actor, parameters, (observations,))

        def one(direction):
            return torch.func.jvp(forward, (parameters,), (direction,))[1]

        return torch.func.vmap(one)(directions)

    return project


def projected_score(logits, tangents, native):
    return projected_coordinate_scores(logits, tangents, native).sum(-1)


def projected_coordinate_scores(logits, tangents, native):
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    da, db = (tangents * logits.sigmoid()).chunk(2, -1)
    total = (alpha + beta).digamma()
    return ((native.log() - alpha.digamma() + total) * da
            + (torch.log1p(-native) - beta.digamma() + total) * db).permute(1, 0, 2)


def counterfactual_contexts(actions, donors):
    """[N,D,D]: row i changes only a_i, marginalizing every other coordinate.

    Donors are fresh independent policy samples, one full action per coordinate.
    Their diagonal supplies the independent reference a_i. Stored a_-i is never
    used, so fresh Monte Carlo donors estimate the other-action marginal.
    """
    mask = torch.eye(actions.shape[-1], device=actions.device, dtype=torch.bool)
    return torch.where(mask, actions[:, None, :], donors)


def counterfactual_advantages(observations, actions, donors, noise, latent_model, reward_model, gamma):
    count, dimensions = actions.shape
    current = counterfactual_contexts(actions, donors)
    states = observations[:, None, :].expand(-1, dimensions, -1).reshape(count * dimensions, -1)
    flat_noise = noise.flatten(0, 1)
    latent = latent_model(states, current.flatten(0, 1), flat_noise)
    reference = latent_model(states, donors.flatten(0, 1), flat_noise)
    # Evaluate nonlinear utility per outcome, not on an averaged latent. Separate
    # action-coordinate interventions produce D distinct advantages, not copies of
    # one scalar target. The actor must detach this frozen model's predictions.
    advantages = (reward_model(latent) - reward_model(reference)).reshape(count, dimensions) / (1 - gamma)
    return advantages, (latent - reference).reshape(count, dimensions, -1)


def pathwise_credit(logits, observations, latent_model, reward_model, noise, gamma):
    """Vector latent costates -> action credit -> Beta-shape logit credit.

    Sampling and its implicit first derivative run on CUDA. Deterministic neural
    utility evaluation is compiled by the caller. No model parameters receive a
    gradient update here, and no sampled scalar return is an actor target.
    """
    logits = logits.detach().requires_grad_(True)
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    distribution = Beta(alpha, beta, validate_args=False)
    native = distribution.rsample()
    latent = latent_model(observations, native, noise)
    # A frozen independent reference action supplies a genuine vector difference
    # in outcome coordinates. Shared noise defines a MODEL coupling, not an
    # identified causal coupling of two physical environment trajectories.
    with torch.no_grad():
        reference_latent = latent_model(observations, distribution.sample(), noise).clone()
    vector_advantage = latent - reference_latent
    # The critic graph never receives a reward-learning update. Pull back a
    # separately evaluated latent cotangent, keeping all coordinates until VJP.
    reward_input = latent.detach().requires_grad_(True)
    latent_credit, = torch.autograd.grad(reward_model(reward_input).sum(), reward_input)
    latent_credit = latent_credit.detach() / (1 - gamma)
    action_credit, = torch.autograd.grad(vector_advantage, native, grad_outputs=latent_credit)
    logit_credit, = torch.autograd.grad(native, logits, grad_outputs=action_credit.detach())
    return logit_credit.detach(), action_credit.detach(), vector_advantage.detach(), latent_credit


def summarize_gate(rows, seed):
    """Paired rollout bootstrap of a model-gradient bias audit, not a variance gate.

    Projections are fixed across rollouts and were never used in model training.
    Confidence is conditional on this checkpoint and these projections, not seeds.
    An uncertain reference produces an inconclusive result, never a promotion.
    """
    reference = np.asarray([row['reference_gradient'] for row in rows], dtype=np.float64)
    model = np.asarray([row['model_gradient'] for row in rows], dtype=np.float64)
    if len(rows) < 16 or not (np.isfinite(reference).all() and np.isfinite(model).all()):
        return {'passed': False, 'status': 'invalid', 'reason': 'incomplete or nonfinite gradient evidence'}
    ref, estimate = reference.mean(0), model.mean(0)
    norm = np.linalg.norm(ref)
    error = np.linalg.norm(estimate - ref)
    standard_error = np.sqrt(reference.var(0, ddof=1).sum() / len(rows))
    snr = norm / max(standard_error, 1e-30)
    indices = np.random.default_rng(seed).integers(len(rows), size=(10000, len(rows)))
    boot_ref, boot_model = reference[indices].mean(1), model[indices].mean(1)
    boot_norm = np.maximum(np.linalg.norm(boot_ref, axis=1), 1e-30)
    relative_upper = float(np.quantile(np.linalg.norm(boot_model - boot_ref, axis=1) / boot_norm, .95))
    cosine = np.sum(boot_model * boot_ref, axis=1) / (boot_norm * np.maximum(np.linalg.norm(boot_model, axis=1), 1e-30))
    cosine_lower = float(np.quantile(cosine, .05))
    reward_mse = sum(row['reward_mse'] for row in rows) / len(rows)
    reward_variance = sum(row['reward_variance'] for row in rows) / len(rows)
    reward_error = reward_mse / max(reward_variance, 1e-30)
    representation_error = float(np.mean([row['representation_normalized_mse'] for row in rows]))
    split = len(reference) // 2
    first, second = reference[:split], reference[split:]
    split_dot = float(first.mean(0) @ second.mean(0))
    rng = np.random.default_rng(seed + 40004)
    first_boot = first[rng.integers(len(first), size=(10000, len(first)))].mean(1)
    second_boot = second[rng.integers(len(second), size=(10000, len(second)))].mean(1)
    split_lower = float(np.quantile((first_boot * second_boot).sum(1), .05))
    usable_reference = snr >= 3 and split_lower > 0
    passed = usable_reference and relative_upper < .5 and cosine_lower > .5 and reward_error < .05 and representation_error < .1
    return {
        'passed': bool(passed), 'status': 'passed' if passed else ('inconclusive' if not usable_reference else 'failed'),
        'criterion': 'reference SNR >=3 and split-mean dot lower95>0; upper95 relative gradient error <0.5; lower95 cosine >0.5; reward NMSE <0.05; representation NMSE <0.1',
        'reference_noise_corrected_norm2': float(norm * norm - standard_error * standard_error),
        'reference_split_mean_dot': split_dot,
        'reference_split_mean_dot_lower95': split_lower,
        'reference_snr': float(snr), 'relative_gradient_error': float(error / max(norm, 1e-30)),
        'relative_gradient_error_upper95': relative_upper, 'gradient_cosine_lower95': cosine_lower,
        'model_reference_norm_ratio': float(np.linalg.norm(estimate) / max(norm, 1e-30)),
        'reward_normalized_mse': float(reward_error),
        'representation_normalized_mse': representation_error,
        'limitations': 'Monte Carlo reference uses a frozen value bootstrap at rollout/time-limit boundaries. This gate cannot prove absence of model exploitation or full-parameter gradient bias.',
    }


def decoder_inputs(name, transitions, latents, context):
    features = transitions if name.startswith('transition') else latents
    return torch.cat((features, context), -1) if 'context' in name else features


def discounted_windows(rewards, next_values, horizon, gamma):
    """All H-step returns; caller must mask reset/rollout crossings identically.

    Every row has both a zero-tail and frozen-value-tail target. This is a
    sensitivity audit, not an assertion that either tail is exact.
    """
    steps, environments = rewards.shape
    kernel = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    padded = F.pad(rewards.T[:, None, :], (0, horizon - 1))
    zero_tail = F.conv1d(padded, kernel[None, None, :]).squeeze(1).T
    indices = (torch.arange(steps, device=rewards.device) + horizon - 1).clamp_max(steps - 1)
    value_tail = zero_tail + gamma ** horizon * next_values[indices]
    return zero_tail, value_tail


def validate_args(args):
    if args.env_id != 'HalfCheetah-v4':
        raise ValueError('This first mechanism experiment is restricted to HalfCheetah-v4')
    if not 0 < args.gamma < 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError('invalid discount or calibration lambda')
    if min(args.num_envs, args.num_steps, args.bellman_steps, args.update_epochs,
           args.noise_dim, args.latent_dim, args.critic_width, args.eval_projections, args.eval_samples,
           args.calibration_rollouts, args.fit_rollouts, args.env_threads) <= 0:
        raise ValueError('positive dimensions, fitting counts, and threads required')
    if args.holdout_rollouts < 16:
        raise ValueError('at least sixteen fresh held-out rollouts required')
    if args.gradient_block_steps <= 0 or args.num_steps % args.gradient_block_steps:
        raise ValueError('gradient blocks must partition each environment rollout')
    if min(args.learning_rate, args.value_learning_rate, args.max_grad_norm) <= 0:
        raise ValueError('positive optimizer settings required')
    return args


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    batch_size = args.num_steps * args.num_envs
    iterations = args.calibration_rollouts + args.fit_rollouts + args.holdout_rollouts
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}'
    run_dir = f'runs/{run_name}'
    resources = ExitStack()
    writer = SummaryWriter(run_dir)
    resources.callback(writer.close)
    try:
        writer.add_text('hyperparameters', '|param|value|\n|-|-|\n' + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()))
        writer.add_text('method', 'Reward-independent joint successor-latent distribution; nonlinear reward decoder; coordinate-counterfactual vector advantages. Actor frozen.')
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        obs_shape = envs.single_observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        agent = Agent(envs).to(device)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        agent.actor.requires_grad_(False)
        actor_start = tuple(p.detach().clone() for p in agent.actor.parameters())
        value_start = ()
        representation = TransitionRepresentation(2 * obs_dim + agent.action_dim + 1, args.latent_dim, args.critic_width).to(device)
        successor = SuccessorCritic(obs_dim, agent.action_dim, args.noise_dim, args.critic_width, args.latent_dim).to(device)
        target_successor = copy.deepcopy(successor).requires_grad_(False)
        reward_decoder = RewardDecoder(successor.latent_dim, args.decoder_width).to(device)
        transition_dim = 2 * obs_dim + agent.action_dim + 1
        small_probe = RewardDecoder(args.latent_dim, 128).to(device)
        probes = nn.ModuleDict({
            'latent_128_unscaled': small_probe,
            'latent_128_scaled': copy.deepcopy(small_probe),
            'transition_256': RewardDecoder(transition_dim, args.decoder_width).to(device),
            'latent_context_256': RewardDecoder(args.latent_dim + args.num_envs, args.decoder_width).to(device),
            'transition_context_256': RewardDecoder(transition_dim + args.num_envs, args.decoder_width).to(device),
        })
        # All probe outputs stay in original normalized-reward units. Scaling
        # residuals improves conditioning without shifting absorbing utility.
        reward_scale = torch.ones((), device=device)
        reward_moments = torch.zeros(3, device=device, dtype=torch.float64)
        context = torch.eye(args.num_envs, device=device).repeat(args.num_steps, 1)
        probe_optimizers = {name: optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
                            for name, model in probes.items()}
        value_optimizer = optim.Adam(agent.critic.parameters(), lr=args.value_learning_rate, eps=1e-5, fused=True)
        successor_optimizer = optim.Adam(successor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        reward_optimizer = optim.Adam(reward_decoder.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        representation_optimizer = optim.Adam(representation.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)

        def statistics(observations):
            logits = agent.actor(observations)
            return logits, agent.get_value(observations).flatten()

        def value_loss(observations, returns, old_values):
            values = agent.get_value(observations).flatten()
            clipped = old_values + (values - old_values).clamp(-.2, .2)
            return .5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()

        def successor_loss(observations, actions, first_noise, second_noise, target):
            first = successor(observations, actions, first_noise)
            second = successor(observations, actions, second_noise)
            return energy_score(first, second, target.detach()).mean()

        def reward_loss(latents, rewards):
            return .5 * ((reward_decoder(latents.detach()) - rewards) / reward_scale).square().mean()

        def make_probe_loss(model, scaled):
            def fit(inputs, rewards):
                error = model(inputs.detach()) - rewards
                if scaled:
                    error = error / reward_scale
                return .5 * error.square().mean()
            return fit

        probe_losses = {name: make_probe_loss(model, not name.endswith('unscaled')) for name, model in probes.items()}
        probe_predictions = {name: model.forward for name, model in probes.items()}

        def counterfactual_model(observations, actions, donors, noise):
            return counterfactual_advantages(observations, actions, donors, noise, successor, reward_decoder, args.gamma)

        value_model = agent.get_value
        policy_model = agent.actor.forward
        target_model = target_successor.forward
        latent_model = successor.forward
        reward_credit_model = reward_decoder.forward
        encode_model = representation.forward
        reconstruct_model = representation.reconstruct
        representation_loss = representation.loss
        reward_model = reward_decoder.forward
        project = make_logit_projector(agent.actor)
        if args.compile:
            statistics = graph_compile(statistics)
            policy_model = graph_compile(policy_model)
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True, options={'triton.cudagraphs': False})
            target_model = graph_compile(target_model)
            reward_model = graph_compile(reward_model)
            value_loss = torch.compile(value_loss, mode=args.compile_mode, fullgraph=True, dynamic=False)
            successor_loss = torch.compile(successor_loss, mode=args.compile_mode, fullgraph=True, dynamic=False)
            reward_loss = torch.compile(reward_loss, mode=args.compile_mode, fullgraph=True, dynamic=False)
            counterfactual_model = graph_compile(counterfactual_model)
            latent_model = torch.compile(latent_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            reward_credit_model = torch.compile(reward_credit_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            encode_model = graph_compile(encode_model)
            reconstruct_model = graph_compile(reconstruct_model)
            representation_loss = torch.compile(representation_loss, mode=args.compile_mode, fullgraph=True, dynamic=False)
            project = torch.compile(project, mode=args.compile_mode, fullgraph=True, dynamic=False)
            probe_losses = {name: torch.compile(fn, mode=args.compile_mode, fullgraph=True, dynamic=False) for name, fn in probe_losses.items()}
            probe_predictions = {name: graph_compile(fn) for name, fn in probe_predictions.items()}
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (b.cpu().numpy() for b in (agent.action_low, agent.action_high))
        sampler = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)
        sampler_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 4]))

        def act(observations):
            native, physical = sampler(host_actor(observations), sampler_rng)
            if not np.isfinite(physical).all():
                raise FloatingPointError('nonfinite actor sample')
            return native, physical.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={'observations': obs_shape, 'native_actions': (agent.action_dim,), 'next_observations': obs_shape})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        terminations = np.empty((args.num_steps, args.num_envs), dtype=bool)
        truncations = np.empty_like(terminations)
        projection_rng = torch.Generator(device=device).manual_seed(args.seed + 20001)
        directions = sample_parameter_directions(agent.actor, args.eval_projections, projection_rng)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
        suppress = np.zeros(args.num_envs, dtype=bool)
        if horizon:
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=lambda obs: act(obs)[1], horizon=horizon,
                                    phase_offsets=compute_phase_offsets(args.num_envs, horizon, args.seed), seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar('timing/warmup_s', time.perf_counter() - start_time, global_step)
        rows, frozen_models = [], []
        interval_start, interval_step = time.perf_counter(), global_step
        for iteration in range(1, iterations + 1):
            stage = ('calibrate' if iteration <= args.calibration_rollouts else
                     'fit' if iteration <= args.calibration_rollouts + args.fit_rollouts else 'holdout')
            if iteration == args.calibration_rollouts + 1:
                rew_norm = FrozenRewardNorm(rew_norm)
                norm_state = {name: torch.from_numpy(getattr(obs_norm, name).copy()) for name in ('means', 'variances', 'counts')}
                norm_state.update(epsilon=obs_norm.epsilon, clip=obs_norm.clip)
                obs_norm = FrozenObsNorm(norm_state, args.num_envs, obs_shape)
                reward_scale.copy_((reward_moments[2] / reward_moments[0] - (reward_moments[1] / reward_moments[0]).square()).clamp_min(1e-12).sqrt())
                agent.critic.requires_grad_(False)
                representation.requires_grad_(False)
                value_start = tuple(p.detach().clone() for p in agent.critic.parameters())
            if iteration == args.calibration_rollouts + args.fit_rollouts + 1:
                successor.requires_grad_(False)
                reward_decoder.requires_grad_(False)
                probes.requires_grad_(False)
                frozen_models = [(model, copy.deepcopy(model.state_dict())) for model in (successor, target_successor, reward_decoder, representation, probes)]
            if stage == 'fit':
                target_successor.load_state_dict(successor.state_dict())
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span('rollout', use_cuda=False):
                    obs_step = next_obs_np
                    native, physical = act(obs_step)
                with timer.span('env', use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(physical)
                with timer.span('normalize_transfer', use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step,
                                  native_actions=native, next_observations=transition_obs)
                    terminations[step], truncations[step] = terms, truncs
                global_step += args.num_envs
                for index, info in enumerate(infos.get('final_info', ())):
                    if info and 'episode' in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        writer.add_scalar('charts/episodic_return', float(info['episode']['r']), global_step)
                        writer.add_scalar('charts/episodic_length', float(info['episode']['l']), global_step)
            with timer.span('gae'), torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                batch = transfer.upload()
                observations = batch.fields['observations'].flatten(0, 1)
                actions = batch.fields['native_actions'].flatten(0, 1)
                next_observations = batch.fields['next_observations'].flatten(0, 1)
                logits, old_values = (t.clone() for t in statistics(observations))
                tail_values = value_model(transfer.observation(next_obs_np)).flatten().clone()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(batch.rewards, old_values.view(args.num_steps, args.num_envs),
                                             batch.terminations, batch.truncations, truncation_values, tail_values,
                                             args.gamma, args.gae_lambda if stage == 'calibrate' else 1.0)
                advantages, returns = advantages.flatten().clone(), returns.flatten().clone()
                transitions = transition_features(observations, actions, next_observations)
                latents = encode_model(transitions).clone()
                if stage == 'calibrate' and iteration > args.calibration_rollouts - min(8, args.calibration_rollouts):
                    reward_moments[0].add_(batch.rewards.numel())
                    reward_moments[1].add_(batch.rewards.double().sum())
                    reward_moments[2].add_(batch.rewards.double().square().sum())
                spans = torch.as_tensor(trajectory_spans(terminations, truncations, args.bellman_steps).flatten(), device=device)
                flat_terms = batch.terminations.flatten().bool()

            def bellman_target():
                with torch.no_grad():
                    sampled_horizon = draw_geometric((batch_size,), args.gamma, device)
                    observed, selected = successor_indices(spans, sampled_horizon, args.num_envs)
                    next_states = next_observations[selected]
                    next_alpha, next_beta = (F.softplus(policy_model(next_states)) + 1).chunk(2, -1)
                    next_actions = Beta(next_alpha, next_beta, validate_args=False).sample()
                    next_noise = torch.randn(batch_size, args.noise_dim, device=device)
                    bootstrapped = target_model(next_states, next_actions, next_noise).clone()
                    return select_successor_target(observed, selected, latents, flat_terms, bootstrapped)

            diagnostics = {'reference/explained_variance': explained_variance(old_values, returns)}
            probe_data = {name: decoder_inputs(name, transitions, latents, context) for name in probes}
            if stage != 'holdout':
                with timer.span('update'):
                    metrics = torch.zeros((args.update_epochs, 8), device=device)
                    probe_metrics = torch.zeros((args.update_epochs, len(probes)), device=device)
                    for epoch in range(args.update_epochs):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        if stage == 'calibrate':
                            loss = value_loss(observations, returns, old_values)
                            value_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            metrics[epoch, 0] = loss.detach()
                            metrics[epoch, 3] = nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm, foreach=True)
                            value_optimizer.step()
                            loss = representation_loss(transitions)
                            representation_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            metrics[epoch, 6] = loss.detach()
                            metrics[epoch, 7] = nn.utils.clip_grad_norm_(representation.parameters(), args.max_grad_norm, foreach=True)
                            representation_optimizer.step()
                        else:
                            target = bellman_target()
                            first_noise, second_noise = torch.randn(2, batch_size, args.noise_dim, device=device).unbind()
                            loss = successor_loss(observations, actions, first_noise, second_noise, target)
                            successor_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            metrics[epoch, 1] = loss.detach()
                            metrics[epoch, 4] = nn.utils.clip_grad_norm_(successor.parameters(), args.max_grad_norm, foreach=True)
                            successor_optimizer.step()
                        if stage == 'fit':
                            loss = reward_loss(latents, batch.rewards.flatten())
                            reward_optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            metrics[epoch, 2] = loss.detach()
                            metrics[epoch, 5] = nn.utils.clip_grad_norm_(reward_decoder.parameters(), args.max_grad_norm, foreach=True)
                            reward_optimizer.step()
                            for probe_index, (name, probe) in enumerate(probes.items()):
                                loss = probe_losses[name](probe_data[name], batch.rewards.flatten())
                                probe_optimizers[name].zero_grad(set_to_none=True)
                                loss.backward()
                                probe_metrics[epoch, probe_index] = loss.detach()
                                nn.utils.clip_grad_norm_(probe.parameters(), args.max_grad_norm, foreach=True)
                                probe_optimizers[name].step()
                        del loss
                    for index, name in enumerate(('reference_value', 'successor_energy', 'reward_mse_half')):
                        diagnostics[f'losses/{name}'] = metrics[:, index].mean()
                    for index, name in enumerate(('reference_value', 'successor', 'reward')):
                        diagnostics[f'grad/{name}_preclip_norm'] = metrics[:, index + 3].mean()
                    diagnostics['losses/representation'] = metrics[:, 6].mean()
                    diagnostics['grad/representation_preclip_norm'] = metrics[:, 7].mean()
                    for probe_index, name in enumerate(probes):
                        diagnostics[f'probes/{name}_fit_loss'] = probe_metrics[:, probe_index].mean()
                    diagnostics['decoder/fixed_reward_scale'] = reward_scale.clone()
            else:
                with timer.span('gradient_diagnostics'):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad():
                        tangents = project(observations, directions).clone()
                        coordinate_scores = projected_coordinate_scores(logits, tangents, actions)
                        reference = coordinate_scores.sum(-1) * advantages[:, None]
                        model = torch.zeros_like(reference)
                        pathwise_model = torch.zeros_like(reference)
                        action_rms = torch.zeros((), device=device)
                        latent_second_moment = torch.zeros((successor.latent_dim, successor.latent_dim), device=device)
                        advantage_second_moment = torch.zeros((agent.action_dim, agent.action_dim), device=device)
                        alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
                    for _ in range(args.eval_samples):
                        with torch.no_grad():
                            donors = Beta(alpha, beta, validate_args=False).sample((agent.action_dim,)).movedim(0, 1).contiguous()
                            counterfactual_noise = torch.randn(batch_size, agent.action_dim, args.noise_dim, device=device)
                            vector_action_advantages, outcome_difference = (x.clone() for x in counterfactual_model(
                                observations, actions, donors, counterfactual_noise))
                            model.add_((coordinate_scores * vector_action_advantages[:, None, :]).sum(-1) / args.eval_samples)
                            flat_difference = outcome_difference.flatten(0, 1)
                            latent_second_moment.add_(flat_difference.T @ flat_difference / (batch_size * agent.action_dim * args.eval_samples))
                            advantage_second_moment.add_(vector_action_advantages.T @ vector_action_advantages / (batch_size * args.eval_samples))
                        noise = torch.randn(batch_size, args.noise_dim, device=device)
                        logit_credit, action_credit, vector_advantage, latent_credit = pathwise_credit(
                            logits, observations, latent_model, reward_credit_model, noise, args.gamma)
                        pathwise_model.add_(torch.einsum('knd,nd->nk', tangents, logit_credit) / args.eval_samples)
                        action_rms.add_(action_credit.square().mean() / args.eval_samples)
                    with torch.no_grad():
                        reference_mean, model_mean = reference.mean(0), model.mean(0)
                        predicted_rewards = reward_model(latents).clone()
                        eigenvalues = torch.linalg.eigvalsh(latent_second_moment).clamp_min(0)
                        # Participation ratio: one for a rank-one vector, up to F
                        # when its second moment is spread equally over F axes.
                        effective_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-30)
                        target = bellman_target()
                        noise1, noise2 = torch.randn(2, batch_size, args.noise_dim, device=device).unbind()
                        predicted1 = latent_model(observations, actions, noise1).clone()
                        predicted2 = latent_model(observations, actions, noise2).clone()
                        collapsed = (predicted1 + predicted2) / 2
                        diagnostics.update({
                            'representation/normalized_mse': (reconstruct_model(latents) - transitions).square().mean() / transitions.var(0, unbiased=False).mean().clamp_min(1e-30),
                            'decoder/reward_mse': (predicted_rewards - batch.rewards.flatten()).square().mean(),
                            'decoder/reward_variance': batch.rewards.flatten().var(unbiased=False),
                            'credit/action_rms': action_rms.sqrt(),
                            'credit/reference_mean_norm': reference_mean.norm(),
                            'credit/model_mean_norm': model_mean.norm(),
                            'credit/gradient_difference_norm': (model_mean - reference_mean).norm(),
                            'credit/pathwise_gradient_difference_norm': (pathwise_model.mean(0) - reference_mean).norm(),
                            'credit/action_advantage_rms': advantage_second_moment.diag().mean().sqrt(),
                            'latent/aggregate_outcome_difference_rank': effective_rank,
                            'latent/advantage_rms': latent_second_moment.diag().mean().sqrt(),
                            'latent/costate_rms': latent_credit.square().mean().sqrt(),
                            'latent/heldout_bellman_energy': energy_score(predicted1, predicted2, target).mean(),
                            'latent/two_sample_mean_energy': energy_score(collapsed, collapsed, target).mean(),
                            'latent/predicted_dispersion': .5 * (predicted1 - predicted2).square().mean(),
                            'latent/target_variance': target.var(0, unbiased=False).mean(),
                            'latent/expected_bootstrap_fraction': (args.gamma ** spans.float()).mean(),
                        })
                        reward_variance = batch.rewards.flatten().var(unbiased=False).clamp_min(1e-30)
                        for name, predict in probe_predictions.items():
                            prediction = predict(probe_data[name]).clone()
                            diagnostics[f'probes/{name}_nmse'] = (prediction - batch.rewards.flatten()).square().mean() / reward_variance
                        reconstruction = reconstruct_model(latents).clone()
                        coordinate_mse = (reconstruction - transitions).square().mean(0)
                        coordinate_variance = transitions.var(0, unbiased=False)
                        for coordinate in range(transition_dim - 1):
                            diagnostics[f'representation/coordinate_{coordinate}_nmse'] = coordinate_mse[coordinate] / coordinate_variance[coordinate].clamp_min(1e-12)
                        diagnostics['representation/real_indicator_mse'] = coordinate_mse[-1]
                        # Separate an observational check from bootstrapped
                        # self-consistency. Restrict starting states to those with
                        # >=512 real transitions before any reset; retain geometric
                        # horizons below 512. The omitted tail mass is gamma**512,
                        # explicitly reported, so this is not an exact full-law test.
                        full_spans = torch.as_tensor(trajectory_spans(terminations, truncations, args.num_steps).flatten(), device=device)
                        # Compare EVERY horizon/tail and the model on exactly the
                        # same states. These omit near-boundary states and do not
                        # replace the full-state reference in the primary gate.
                        same_states = (full_spans >= 512).float()
                        same_count = same_states.sum().clamp_min(1)
                        score = coordinate_scores.sum(-1)
                        next_values = value_model(next_observations).flatten().clone().view(args.num_steps, args.num_envs)
                        sensitivity = {'model': ((model * same_states[:, None]).sum(0) / same_count).cpu().tolist()}
                        for return_horizon in (64, 256, 512):
                            zero_return, value_return = discounted_windows(batch.rewards, next_values, return_horizon, args.gamma)
                            for tail_name, reference_return in (('zero', zero_return), ('value', value_return)):
                                projected = score * (reference_return.flatten() - old_values)[:, None]
                                sensitivity[f'h{return_horizon}_{tail_name}'] = ((projected * same_states[:, None]).sum(0) / same_count).cpu().tolist()
                        immediate_gradient = (score * batch.rewards.flatten()[:, None]).mean(0)
                        horizon_sample = draw_geometric((batch_size,), args.gamma, device)
                        usable = (full_spans >= 512) & (horizon_sample < 512)
                        _, observed_indices = successor_indices(full_spans, horizon_sample, args.num_envs)
                        observed_target = latents[observed_indices]
                        weight = usable.float()
                        count = weight.sum().clamp_min(1)
                        diagnostics.update({
                            'latent/observed_future_energy': (energy_score(predicted1, predicted2, observed_target) * weight).sum() / count,
                            'latent/observed_mean_energy': (energy_score(collapsed, collapsed, observed_target) * weight).sum() / count,
                            'latent/observed_future_samples': weight.sum(),
                            'latent/observed_tail_mass_omitted': torch.as_tensor(args.gamma ** 512, device=device),
                        })
                        # Temporal blocks retain within-environment dependence; these
                        # are diagnostics, NOT replacements for the mean/bias gate.
                        for name, projected in (('reference', reference), ('model', model)):
                            blocks = projected.reshape(-1, args.gradient_block_steps, args.num_envs, args.eval_projections).mean(1).flatten(0, 1)
                            diagnostics[f'credit/{name}_block_variance'] = blocks.var(0, unbiased=False).mean()
                        row = {'rollout': len(rows) + 1, 'step': global_step,
                               'reference_gradient': reference_mean.cpu().tolist(), 'model_gradient': model_mean.cpu().tolist(),
                               'immediate_gradient': immediate_gradient.cpu().tolist(),
                               'delayed_gradient': (reference_mean - immediate_gradient).cpu().tolist(),
                               'same_state_sensitivity': sensitivity,
                               'same_state_count': float(same_count.cpu()),
                               'decoder_probes': {name: float(diagnostics[f'probes/{name}_nmse'].cpu()) for name in probes}}
                        del tangents, reference, model, pathwise_model, coordinate_scores, logit_credit, action_credit, vector_advantage, latent_credit
            logged = gather_metrics(diagnostics)
            if any(not np.isfinite(value) for name, value in logged.items() if not name.endswith('explained_variance')):
                raise FloatingPointError('nonfinite predictive critic or gradient evidence')
            if stage == 'holdout':
                row.update(reward_mse=logged['decoder/reward_mse'], reward_variance=logged['decoder/reward_variance'])
                row['representation_normalized_mse'] = logged['representation/normalized_mse']
                rows.append(row)
                with open(f'{run_dir}/holdout.json', 'w') as output:
                    json.dump(rows, output, indent=2, allow_nan=False)
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar('charts/SPS', global_step / (now - start_time), global_step)
            writer.add_scalar('charts/interval_SPS', (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f'timing/{phase}_s', timing['total_s'], global_step)
            timer.reset()
            writer.flush()
            print(f'stage={stage} rollout={iteration}/{iterations} step={global_step} metrics={json.dumps(logged)}', flush=True)
            interval_start, interval_step = now, global_step
        if not all(torch.equal(p, q) for p, q in zip(agent.actor.parameters(), actor_start)):
            raise RuntimeError('fixed-policy actor changed')
        if not all(torch.equal(p, q) for p, q in zip(agent.critic.parameters(), value_start)):
            raise RuntimeError('reference value changed after calibration')
        for model, frozen in frozen_models:
            if not all(torch.equal(value, frozen[key]) for key, value in model.state_dict().items()):
                raise RuntimeError('model changed during holdout')
        report = summarize_gate(rows, args.seed)
        report.update(args=vars(args), fresh_initialization=True, checkpoint_loaded=False, actor_unchanged=True, models_frozen_for_holdout=True, holdout=rows,
                      latent_definition='learned reward-independent transition encoder; full-transition reconstruction plus covariance regularization; frozen before successor fitting',
                      critic_supervision='joint distributional Bellman energy score; no reward/return gradient',
                      primary_estimator='coordinate-counterfactual advantages from nonlinear decoded future-latent samples; pathwise estimator diagnostic only')
        with open(f'{run_dir}/gate.json', 'w') as output:
            json.dump(report, output, indent=2, allow_nan=False)
        writer.add_scalar('gate/passed', int(report['passed']), global_step)
        if args.save_model:
            torch.save({'successor': successor.state_dict(), 'target_successor': target_successor.state_dict(),
                        'actor': agent.actor.state_dict(),
                        'obs_norm': {name: getattr(obs_norm, name) for name in ('means', 'variances', 'counts', 'epsilon', 'clip')},
                        'decoder_probes': probes.state_dict(),
                        'representation': representation.state_dict(),
                        'reward_decoder': reward_decoder.state_dict(), 'reference_value': agent.critic.state_dict(),
                        'args': vars(args), 'reward_norm': {name: getattr(rew_norm, name) for name in ('means', 'variances', 'counts', 'epsilon', 'clip')}},
                       f'{run_dir}/{args.exp_name}.cleanrl_model')
        print('GATE_RESULT=' + json.dumps({k: v for k, v in report.items() if k not in {'args', 'holdout'}}), flush=True)
        print(f'gate report saved to {run_dir}/gate.json', flush=True)
    finally:
        resources.close()


if __name__ == '__main__':
    main()
