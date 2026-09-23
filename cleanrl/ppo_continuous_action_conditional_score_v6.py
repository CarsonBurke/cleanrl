# Conditional-score critic v6: fresh, decoder-free state-action credit.
# Predict a 2-direction Beta score latent per actuator from REAL future outcomes.
# Vector action-statistic supervision; measured reward weights vectors only after
# inference. No generated outcomes, scalar value critic, GAE, or value bootstrap.
# Audit the exact finite-episode gradient with discounted age weighting before RL.
# Full physical-outcome conditioning is primary; reduced conditioning is an ablation.
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
from cleanrl.shared.ppo_loop import gather_metrics
from cleanrl.shared.norm_residual import make_norm_residual_trunk
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

NATIVE_TASKS = frozenset(('HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4'))
HORIZON_LOWER = (1, 2, 4, 8, 16, 32, 64, 128, 256)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    """Freeze calibrated observation statistics for fitting and evaluation.

    Shared VectorObsNorm remains the training implementation. Ping-pong outputs
    preserve the old observation until RolloutTransfer stages the same transition.
    """
    def __init__(self, state, num_envs, obs_shape):
        self.means = state["means"].cpu().numpy().copy()
        self.variances = state["variances"].cpu().numpy().copy()
        self.counts = state["counts"].cpu().numpy().copy()
        self.epsilon, self.clip = state["epsilon"], state["clip"]
        if self.means.shape != (num_envs, int(np.prod(obs_shape))):
            raise ValueError("frozen statistics must match environment count and observation shape")
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


def transition_features(observations, actions, next_observations):
    """Observed physical outcome: state, future action, displacement, indicator."""
    return torch.cat((observations, 2 * actions - 1, (next_observations - observations) / math.sqrt(2),
                      torch.ones_like(observations[:, :1])), -1)


def make_logit_projector(actor):
    parameters = dict(actor.named_parameters())

    def project(observations, directions):
        def forward(parameters):
            return torch.func.functional_call(actor, parameters, (observations,))

        def one(direction):
            return torch.func.jvp(forward, (parameters,), (direction,))[1]

        return torch.func.vmap(one)(directions)

    return project


@dataclass
class Args:
    exp_name: str = 'conditional_score_v6_fresh'
    seed: int = 1
    env_id: str = 'HalfCheetah-v4'
    num_envs: int = 16
    num_steps: int = 2048
    gamma: float = .99
    calibration_rollouts: int = 32
    fit_rollouts: int = 128
    holdout_rollouts: int = 256
    update_epochs: int = 10
    minibatch_size: int = 8192
    critic_width: int = 256
    learning_rate: float = 3e-4
    max_grad_norm: float = .5
    eval_projections: int = 32
    env_backend: str = 'auto'
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = 'reduce-overhead'
    capture_video: bool = False
    torch_deterministic: bool = True
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    save_model: bool = True


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError('Beta actor requires a Box action space')
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError('finite ordered action bounds required')
        self.register_buffer('action_low', torch.as_tensor(low.reshape(-1).copy()))
        self.register_buffer('action_high', torch.as_tensor(high.reshape(-1).copy()))
        self.actor = nn.Sequential(
            make_norm_residual_trunk(int(np.prod(envs.single_observation_space.shape)), 64,
                                    placement='pre', norm_kind='rms', activation='stiglu'),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=.01))


def beta_geometry(logits, actions):
    """F=L L^T, whitened scores z=L^-1(T-E[T]), and policy parameters.

    Layout is [batch, action, alpha/beta]. Work in FP64 for trigamma subtraction,
    then return the input dtype. No Fisher damping changes the target geometry.
    """
    alpha, beta = (F.softplus(logits.double()) + 1).chunk(2, -1)
    total = alpha + beta
    off = -torch.polygamma(1, total)
    l11 = (torch.polygamma(1, alpha) + off).sqrt()
    l21 = off / l11
    l22 = (torch.polygamma(1, beta) + off - l21.square()).sqrt()
    score_a = actions.double().log() - alpha.digamma() + total.digamma()
    score_b = torch.log1p(-actions.double()) - beta.digamma() + total.digamma()
    z1 = score_a / l11
    z2 = (score_b - l21 * z1) / l22
    latent = torch.stack((z1, z2), -1).to(logits.dtype)
    factor = torch.stack((l11, l21, l22), -1).to(logits.dtype)
    parameters = torch.stack((alpha, beta), -1).to(logits.dtype)
    return latent, factor, parameters


def unwhiten(latent, factor):
    first = factor[..., 0] * latent[..., 0]
    second = factor[..., 1] * latent[..., 0] + factor[..., 2] * latent[..., 1]
    return torch.stack((first, second), -1)


def score_tangents(logits, tangents, factor):
    """Contract parameter directions with J_eta^T L, preserving both directions."""
    da, db = (tangents * logits.sigmoid()).chunk(2, -1)
    return torch.stack((da * factor[..., 0] + db * factor[..., 1], db * factor[..., 2]), -1)


def project_latent(latent, tangents):
    return torch.einsum('ndk,pndk->np', latent, tangents)


def actor_surrogate(logits, natural_credit):
    """Local VJP objective only; not an off-policy or multi-epoch PPO identity."""
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    return (torch.stack((alpha - 1, beta - 1), -1) * natural_credit.detach()).sum((-1, -2)).mean()


def horizon_features(horizons):
    t = horizons.float()[..., None]
    scales = torch.tensor((1., 2., 4., 8., 16., 32., 64., 128., 256.), device=t.device)
    return torch.cat((torch.log1p(t) / math.log(1001), torch.exp(-t / scales)), -1)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x):
        return x + self.second(F.silu(self.first(self.norm(x)))) / math.sqrt(2)


class ConditionalScoreCritic(nn.Module):
    """Reward-conditioned, action-statistic-supervised vector credit.

    The reduced-outcome ablation retains starting state, age, source-policy
    parameters, horizon, reward, and normalization context. It drops only the
    physical future transition. Neither model receives the source sampled action.
    """
    def __init__(self, observation_dim, action_dim, num_envs, width=256, physical_outcome=True):
        super().__init__()
        self.physical_outcome = physical_outcome
        self.action_dim = action_dim
        self.environment = nn.Embedding(num_envs, 8)
        outcome_dim = 2 * observation_dim + action_dim + 1 if physical_outcome else 0
        input_dim = observation_dim + 2 * action_dim + 1 + 10 + 1 + 8 + outcome_dim
        self.trunk = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                                   ResidualBlock(width), ResidualBlock(width))
        # Each actuator has a learned 32D latent and a shared 2D score readout.
        # Joint context allows correlated action effects even though pi factorizes.
        self.action_latents = nn.Linear(width, action_dim * 32)
        self.score_readout = layer_init(nn.Linear(32, 2), std=.01)

    def forward(self, state, policy_parameters, age, environment, outcome, reward, horizon):
        context = [state, policy_parameters.log().flatten(1), age[:, None].float() / 1000,
                   horizon_features(horizon), reward[:, None], self.environment(environment)]
        if self.physical_outcome:
            context.append(outcome)
        latent = F.silu(self.action_latents(self.trunk(torch.cat(context, -1))))
        return self.score_readout(latent.view(-1, self.action_dim, 32))


def complete_episode_layout(ages, terminations, truncations, episode_length):
    """Select whole fixed-length episodes, never action-dependent partial windows.

    Unknown warmup ages are -1 until the next reset. The first implementation
    deliberately rejects early termination: its selection proof is HalfCheetah-
    specific and must not silently extend to variable-length episodes.
    """
    if terminations.any():
        raise ValueError('this fixed-length gate cannot censor early terminations')
    steps, environments = ages.shape
    if np.any(truncations & (ages >= 0) & (ages != episode_length - 1)):
        raise ValueError('unexpected episode length in the fixed-horizon gate')
    ids = np.full_like(ages, -1, dtype=np.int64)
    episode_count = 0
    for env in range(environments):
        for end in np.flatnonzero(truncations[:, env]):
            start = end - episode_length + 1
            if start >= 0 and ages[start, env] == 0:
                if not np.array_equal(ages[start:end + 1, env], np.arange(episode_length)):
                    raise ValueError('episode age/order mismatch')
                ids[start:end + 1, env] = episode_count
                episode_count += 1
    return ids, episode_count


def sample_band(remaining, band, gamma):
    """One delayed h per stratum, with its exact finite-episode discount mass.

    remaining includes h=0. Empty strata have zero weight and safe h=0.
    In a nonempty [lo,hi) band, q(h)=gamma^h / mass.
    """
    lower = torch.tensor(HORIZON_LOWER, device=remaining.device)
    upper = torch.cat((lower[1:], torch.full_like(lower[:1], 1000000000)))
    lo = lower[band]
    width = (torch.minimum(remaining, upper[band]) - lo).clamp_min(0)
    probability = -torch.expm1(width.double() * math.log(gamma))
    u = torch.rand(remaining.shape, device=remaining.device, dtype=torch.float64)
    offset = (torch.log1p(-u * probability) / math.log(gamma)).floor().long()
    offset = torch.minimum(offset, (width - 1).clamp_min(0))
    horizon = torch.where(width > 0, lo + offset, torch.zeros_like(remaining))
    mass = gamma ** lo.double() * probability / (1 - gamma)
    return horizon, mass.float()


def discounted_episode_returns(rewards, episode_ids, gamma):
    """Exact observed finite-episode returns; invalid rows contribute zero."""
    # Parallel doubling scan: O(log T) batched operations, no timestep kernel loop.
    result = rewards * (episode_ids >= 0)
    offset = 1
    while offset < rewards.shape[0]:
        same = (episode_ids[:-offset] >= 0) & (episode_ids[:-offset] == episode_ids[offset:])
        result = torch.cat((result[:-offset] + gamma ** offset * result[offset:] * same, result[-offset:]), 0)
        offset *= 2
    return result


def episode_moments(contributions, episode_ids, count):
    valid = episode_ids >= 0
    sums = torch.zeros(count, contributions.shape[-1], device=contributions.device, dtype=torch.float64)
    sums.index_add_(0, episode_ids[valid], contributions[valid].double())
    return sums.mean(0), sums.square().sum(), sums


def compare_gradients(reference, model, rng, bootstrap_indices, counts=None):
    counts = np.ones(len(reference)) if counts is None else np.asarray(counts, dtype=np.float64)
    def weighted_mean(values, weights):
        return (values * weights[..., None]).sum(-2) / weights.sum(-1)[..., None]
    ref, estimate = weighted_mean(reference, counts), weighted_mean(model, counts)
    norm = np.linalg.norm(ref)
    # Cluster-robust standard error of the episode-weighted mean; each rollout
    # remains intact in the bootstrap, including all its environments/episodes.
    residual = (reference - ref) * counts[:, None]
    se = np.sqrt(len(reference) / (len(reference) - 1) * (residual * residual).sum()) / counts.sum()
    br = weighted_mean(reference[bootstrap_indices], counts[bootstrap_indices])
    bm = weighted_mean(model[bootstrap_indices], counts[bootstrap_indices])
    bn = np.linalg.norm(br, axis=1)
    relative = np.linalg.norm(bm - br, axis=1) / np.maximum(bn, 1e-30)
    cosine = (br * bm).sum(1) / np.maximum(bn * np.linalg.norm(bm, axis=1), 1e-30)
    split = len(reference) // 2
    first, second = reference[:split], reference[split:]
    i = rng.integers(len(first), size=(4096, len(first)))
    j = rng.integers(len(second), size=(4096, len(second)))
    bfirst = weighted_mean(first[i], counts[:split][i])
    bsecond = weighted_mean(second[j], counts[split:][j])
    return dict(reference_snr=float(norm / max(se, 1e-30)),
                reference_split_dot_lower95=float(np.quantile((bfirst * bsecond).sum(1), .05)),
                relative_gradient_error=float(np.linalg.norm(estimate - ref) / max(norm, 1e-30)),
                relative_gradient_error_upper95=float(np.quantile(relative, .95)),
                cosine=float(np.dot(ref, estimate) / max(norm * np.linalg.norm(estimate), 1e-30)),
                cosine_lower95=float(np.quantile(cosine, .05)),
                norm_ratio=float(np.linalg.norm(estimate) / max(norm, 1e-30)))


def summarize_gate(rows, seed):
    if len(rows) < 16:
        raise ValueError('at least 16 independent held-out rollouts required')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(rows), size=(4096, len(rows)))
    counts = np.asarray([r['episodes'] for r in rows], dtype=np.float64)
    means = {name: np.asarray([r[name + '_gradient'] for r in rows], dtype=np.float64)
             for name in ('reference', 'sampled_reference', 'physical', 'reduced')}
    squares = {name: np.asarray([r[name + '_episode_square_sum'] for r in rows], dtype=np.float64)
               for name in means}
    if not all(np.isfinite(value).all() for value in (*means.values(), *squares.values())):
        raise ValueError('nonfinite gate evidence')

    def variance(name, selection):
        n = counts[selection].sum(-1)
        total = (means[name][selection] * counts[selection][..., None]).sum(-2)
        return (squares[name][selection].sum(-1) - (total * total).sum(-1) / n) / (n - 1)

    all_indices = np.arange(len(rows))
    reference_variance = variance('reference', all_indices)
    ref_boot = variance('reference', indices)
    result = {}
    for name in ('physical', 'reduced', 'sampled_reference'):
        comparison = compare_gradients(means['reference'], means[name], rng, indices, counts)
        ratios = variance(name, indices) / np.maximum(ref_boot, 1e-30)
        br = (means['reference'][indices] * counts[indices][..., None]).sum(1) / counts[indices].sum(1)[:, None]
        bm = (means[name][indices] * counts[indices][..., None]).sum(1) / counts[indices].sum(1)[:, None]
        norm_ratios_squared = (bm * bm).sum(1) / np.maximum((br * br).sum(1), 1e-30)
        normalized_ratios = ratios / np.maximum(norm_ratios_squared, 1e-30)
        comparison.update(episode_variance_ratio=float(variance(name, all_indices) / max(reference_variance, 1e-30)),
                          episode_variance_ratio_upper95=float(np.quantile(ratios, .95)),
                          signal_normalized_variance_ratio=float(variance(name, all_indices) / max(reference_variance * comparison['norm_ratio'] ** 2, 1e-30)),
                          signal_normalized_variance_ratio_upper95=float(np.quantile(normalized_ratios, .95)))
        usable = comparison['reference_snr'] >= 3 and comparison['reference_split_dot_lower95'] > 0
        mean_pass = comparison['relative_gradient_error_upper95'] < .5 and comparison['cosine_lower95'] > .5
        variance_pass = (comparison['episode_variance_ratio_upper95'] < .95
                         and comparison['signal_normalized_variance_ratio_upper95'] < .95)
        comparison.update(passed=bool(usable and mean_pass and variance_pass),
                          status='inconclusive' if not usable else ('passed' if mean_pass and variance_pass else 'failed'))
        result[name] = comparison
    return dict(passed=result['physical']['passed'], status=result['physical']['status'], comparisons=result,
                criterion='Primary physical-outcome critic: reference SNR>=3, split-dot lower95>0, relative error upper95<0.5, cosine lower95>0.5, complete-episode variance ratio upper95<0.95 AND variance ratio divided by squared gradient-norm ratio upper95<0.95 (reject trivial shrinkage).',
                limitations='Frozen fresh random policy; projected gradients; finite 1000-step discounted episode objective. No continuing-task tail, policy-training, or global-optimality claim. Reduced-outcome critic is exploratory.')


def validate_args(args):
    if args.env_id != 'HalfCheetah-v4':
        raise ValueError('The complete-episode selection proof requires fixed-length HalfCheetah')
    if not 0 < args.gamma < 1:
        raise ValueError('discount must be in (0,1)')
    if min(args.num_envs, args.env_threads, args.calibration_rollouts, args.fit_rollouts,
           args.update_epochs, args.minibatch_size, args.critic_width, args.eval_projections) <= 0:
        raise ValueError('positive dimensions and fitting counts required')
    if args.num_steps < 2000 or args.holdout_rollouts < 16:
        raise ValueError('need at least two episode lengths per rollout and 16 holdout rollouts')
    if min(args.learning_rate, args.max_grad_norm) <= 0:
        raise ValueError('positive optimizer settings required')
    return args


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    iterations = args.calibration_rollouts + args.fit_rollouts + args.holdout_rollouts
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{time.time_ns()}'
    run_dir = f'runs/{run_name}'
    with ExitStack() as resources:
        writer = SummaryWriter(run_dir)
        resources.callback(writer.close)
        writer.add_text('hyperparameters', '|param|value|\n|-|-|\n' + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()))
        writer.add_text('method', 'Fresh frozen Beta actor; conditional-score vector critic; actual rewards only; exact complete-episode reference; no GAE, value network, generated outcome, or reward decoder.')
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        obs_shape = envs.single_observation_space.shape
        obs_dim = int(np.prod(obs_shape))
        length = episode_horizon(args.env_id)
        if length != 1000:
            raise ValueError('this registered finite-episode gate expects 1000-step episodes')
        agent = Agent(envs).to(device)
        agent.actor.requires_grad_(False)
        actor_start = copy.deepcopy(agent.actor.state_dict())
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        # Shared reward normalization is logged; objective uses RAW reward divided
        # by ONE global fixed scale, avoiding per-env reweighting and clipping.
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        critics = {name: ConditionalScoreCritic(obs_dim, agent.action_dim, args.num_envs,
                                               args.critic_width, physical_outcome=(name == 'physical')).to(device)
                   for name in ('physical', 'reduced')}
        optimizers = {name: optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
                      for name, model in critics.items()}

        def build_loss(model):
            def loss(state, parameters, age, environment, outcome, reward, horizon, target, valid):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    predicted = model(state, parameters, age, environment, outcome, reward, horizon)
                per_row = .5 * (predicted.float() - target.detach()).square().mean((-1, -2))
                return (per_row * valid).sum() / valid.sum().clamp_min(1)
            return loss

        def build_predict(model):
            def predict(state, parameters, age, environment, outcome, reward, horizon):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return model(state, parameters, age, environment, outcome, reward, horizon).float()
            return predict

        loss_functions = {name: build_loss(model) for name, model in critics.items()}
        predictors = {name: build_predict(model) for name, model in critics.items()}
        policy_model = agent.actor.forward
        project = make_logit_projector(agent.actor)
        returns_fn = discounted_episode_returns
        geometry_fn = beta_geometry
        if args.compile:
            policy_model = graph_compile(policy_model)
            project = torch.compile(project, fullgraph=True, mode=args.compile_mode)
            returns_fn = torch.compile(returns_fn, fullgraph=True, mode=args.compile_mode)
            geometry_fn = torch.compile(geometry_fn, fullgraph=True, mode=args.compile_mode)
            loss_functions = {name: torch.compile(fn, fullgraph=True, mode=args.compile_mode)
                              for name, fn in loss_functions.items()}
            predictors = {name: graph_compile(fn) for name, fn in predictors.items()}
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        sampler = make_beta_sampler(args.num_envs, agent.action_dim,
                                    agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy())
        sampler_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 6]))

        def act(observations):
            native, physical = sampler(host_actor(observations), sampler_rng)
            if not np.isfinite(physical).all():
                raise FloatingPointError('nonfinite actor sample')
            return native, physical.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                  non_blocking=args.non_blocking_transfers,
                                  fields={'observations': obs_shape, 'native_actions': (agent.action_dim,),
                                          'next_observations': obs_shape, 'raw_rewards': ()})
        resources.callback(transfer.close)
        ages = np.empty((args.num_steps, args.num_envs), dtype=np.int64)
        terms_buffer = np.empty_like(ages, dtype=bool)
        truncs_buffer = np.empty_like(ages, dtype=bool)
        raw_rewards_buffer = np.empty_like(ages, dtype=np.float64)
        current_age = np.full(args.num_envs, -1, dtype=np.int64)
        calibration_sum = np.zeros(length)
        calibration_square = np.zeros(length)
        calibration_count = np.zeros(length, dtype=np.int64)
        reward_scale = 1.
        reward_center = torch.zeros(length, device=device)
        projection_rng = torch.Generator(device=device).manual_seed(args.seed + 20001)
        directions = sample_parameter_directions(agent.actor, args.eval_projections, projection_rng)
        rows_index = torch.arange(args.num_steps * args.num_envs, device=device)
        environment_index = rows_index % args.num_envs
        timer = PhaseTimer()
        started = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        if args.staggered_starts and args.num_envs > 1:
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=lambda obs: act(obs)[1], horizon=length,
                                    phase_offsets=compute_phase_offsets(args.num_envs, length, args.seed), seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
            current_age.fill(0)
        rows, frozen = [], {}
        interval_start, interval_step = time.perf_counter(), global_step
        for iteration in range(1, iterations + 1):
            stage = ('calibrate' if iteration <= args.calibration_rollouts else
                     'fit' if iteration <= args.calibration_rollouts + args.fit_rollouts else 'holdout')
            if iteration == args.calibration_rollouts + 1:
                if (calibration_count == 0).any():
                    raise ValueError('calibration missed episode ages')
                total = calibration_count.sum()
                reward_scale = float(np.sqrt(calibration_square.sum() / total - (calibration_sum.sum() / total) ** 2))
                if not np.isfinite(reward_scale) or reward_scale <= 0:
                    raise ValueError('degenerate raw reward scale')
                reward_center.copy_(torch.as_tensor(calibration_sum / calibration_count / reward_scale, device=device))
                norm_state = {name: torch.from_numpy(getattr(obs_norm, name).copy()) for name in ('means', 'variances', 'counts')}
                norm_state.update(epsilon=obs_norm.epsilon, clip=obs_norm.clip)
                obs_norm = FrozenObsNorm(norm_state, args.num_envs, obs_shape)
                print(f'FROZEN_CALIBRATION raw_reward_scale={reward_scale}', flush=True)
            if iteration == args.calibration_rollouts + args.fit_rollouts + 1:
                frozen = {name: copy.deepcopy(model.state_dict()) for name, model in critics.items()}
                for model in critics.values():
                    model.requires_grad_(False)
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span('rollout', use_cuda=False):
                    obs_step = next_obs_np
                    native, physical = act(obs_step)
                    ages[step] = current_age
                with timer.span('env', use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(physical)
                with timer.span('normalize_transfer', use_cuda=False):
                    normalized_reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, physical_next = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, normalized_reward, terms, truncs, observations=obs_step,
                                  native_actions=native, next_observations=physical_next, raw_rewards=raw_reward)
                    raw_rewards_buffer[step] = raw_reward
                    terms_buffer[step], truncs_buffer[step] = terms, truncs
                    current_age = np.where(terms | truncs, 0, np.where(current_age >= 0, current_age + 1, -1))
                global_step += args.num_envs
                for index, info in enumerate(infos.get('final_info', ())):
                    if info and 'episode' in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        writer.add_scalar('charts/episodic_return', float(info['episode']['r']), global_step)
                        writer.add_scalar('charts/episodic_length', float(info['episode']['l']), global_step)

            diagnostics = {}
            if stage == 'calibrate':
                known = ages >= 0
                phase = ages[known]
                if np.any(phase >= length):
                    raise ValueError('unrecognized episode horizon')
                calibration_count += np.bincount(phase, minlength=length)
                calibration_sum += np.bincount(phase, weights=raw_rewards_buffer[known], minlength=length)
                calibration_square += np.bincount(phase, weights=raw_rewards_buffer[known] ** 2, minlength=length)
                diagnostics['calibration/raw_reward_mean'] = float(raw_rewards_buffer.mean())
            else:
                with timer.span('prepare'), torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    batch = transfer.upload()
                    observations = batch.fields['observations'].flatten(0, 1)
                    actions = batch.fields['native_actions'].flatten(0, 1)
                    next_observations = batch.fields['next_observations'].flatten(0, 1)
                    raw_rewards = batch.fields['raw_rewards'].flatten() / reward_scale
                    age = torch.as_tensor(ages.copy().flatten(), device=device)
                    layout, episode_count = complete_episode_layout(ages, terms_buffer, truncs_buffer, length)
                    if episode_count == 0:
                        raise ValueError('no fully observed episodes')
                    ids = torch.as_tensor(layout.flatten(), device=device)
                    valid = ids >= 0
                    indices = valid.nonzero().flatten()
                    remaining = torch.where(valid, length - age, 1)
                    centered_rewards = raw_rewards - reward_center[age.clamp_min(0)]
                    logits = policy_model(observations).clone()
                    latent, factor, parameters = (x.clone() for x in geometry_fn(logits, actions))
                    outcomes = transition_features(observations, actions, next_observations)
                    discounts = args.gamma ** age.float() * valid
                    diagnostics['data/complete_episodes'] = float(episode_count)
                    diagnostics['data/fraction_used'] = valid.float().mean()
                    diagnostics['normalization/raw_reward_scale'] = reward_scale
                    diagnostics['latent/target_second_moment'] = latent[valid].square().mean()

                if stage == 'fit':
                    with timer.span('update'):
                        steps = args.update_epochs * math.ceil(episode_count * length / args.minibatch_size)
                        metrics = torch.zeros(steps, 4, device=device)
                        for update in range(steps):
                            if args.compile:
                                torch.compiler.cudagraph_mark_step_begin()
                            chosen = indices[torch.randint(indices.numel(), (args.minibatch_size,), device=device)]
                            band = torch.randint(len(HORIZON_LOWER), (args.minibatch_size,), device=device)
                            h, mass = sample_band(remaining[chosen], band, args.gamma)
                            selected = chosen + h * args.num_envs
                            inputs = (observations[chosen], parameters[chosen], age[chosen], environment_index[chosen],
                                      outcomes[selected], raw_rewards[selected], h, latent[chosen], mass > 0)
                            for column, (name, model) in enumerate(critics.items()):
                                loss = loss_functions[name](*inputs)
                                optimizers[name].zero_grad(set_to_none=True)
                                loss.backward()
                                metrics[update, column] = loss.detach()
                                metrics[update, column + 2] = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, foreach=True)
                                optimizers[name].step()
                                del loss
                        for column, name in enumerate(critics):
                            diagnostics[f'losses/{name}_vector_mse_half'] = metrics[:, column].mean()
                            diagnostics[f'grad/{name}_preclip_norm'] = metrics[:, column + 2].mean()
                else:
                    with timer.span('update'), torch.no_grad():
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        tangents = score_tangents(logits, project(observations, directions).clone(), factor)
                        score = project_latent(latent, tangents)
                        returns = returns_fn(centered_rewards.view(args.num_steps, args.num_envs),
                                             ids.view(args.num_steps, args.num_envs), args.gamma).flatten().clone()
                        credits = {name: centered_rewards[:, None, None] * latent for name in ('sampled_reference', *critics)}
                        band_diagnostics = {}
                        for band_index in range(len(HORIZON_LOWER)):
                            band = torch.full_like(age, band_index)
                            h, mass = sample_band(remaining, band, args.gamma)
                            selected = rows_index + h * args.num_envs
                            weight = centered_rewards[selected] * mass
                            credits['sampled_reference'].add_(latent * weight[:, None, None])
                            inputs = (observations, parameters, age, environment_index, outcomes[selected], raw_rewards[selected], h)
                            active = valid & (mass > 0)
                            for name in critics:
                                predicted = predictors[name](*inputs).clone()
                                credits[name].add_(predicted * weight[:, None, None])
                                mse = (predicted - latent).square().mean((-1, -2))
                                band_diagnostics[f'{name}/band_{HORIZON_LOWER[band_index]}_mse'] = (mse * active).sum() / active.sum().clamp_min(1)
                                band_diagnostics[f'{name}/band_{HORIZON_LOWER[band_index]}_prediction_energy'] = (predicted.square().mean((-1, -2)) * active).sum() / active.sum().clamp_min(1)

                        contributions = {'reference': score * (returns * discounts)[:, None]}
                        for name, credit in credits.items():
                            contributions[name] = project_latent(credit, tangents) * discounts[:, None]
                        row = {'rollout': iteration, 'step': global_step, 'episodes': episode_count, 'horizons': {}}
                        for name, value in contributions.items():
                            mean, squared, _ = episode_moments(value, ids, episode_count)
                            row[name + '_gradient'] = mean.cpu().tolist()
                            row[name + '_episode_square_sum'] = float(squared.cpu())
                            diagnostics[f'credit/{name}_mean_norm'] = mean.norm()
                        # Exact-horizon diagnostics use the same source states in
                        # each pair. No fitted tail and no sampled reward forecast.
                        for horizon in (1, 4, 16, 64):
                            active = valid & (remaining > horizon)
                            h = torch.where(active, torch.full_like(age, horizon), 0)
                            selected = rows_index + h * args.num_envs
                            weight = centered_rewards[selected] * discounts * active
                            inputs = (observations, parameters, age, environment_index, outcomes[selected], raw_rewards[selected], h)
                            horizon_row = {'reference': (score * weight[:, None]).sum(0).cpu().tolist()}
                            for name in critics:
                                prediction = predictors[name](*inputs).clone()
                                estimate = project_latent(prediction, tangents) * weight[:, None]
                                horizon_row[name] = estimate.sum(0).cpu().tolist()
                            row['horizons'][str(horizon)] = {name: (np.asarray(value) / episode_count).tolist()
                                                            for name, value in horizon_row.items()}
                        diagnostics.update(band_diagnostics)
                        rows.append(row)
                        with open(f'{run_dir}/holdout.json', 'w') as output:
                            json.dump(rows, output, indent=2, allow_nan=False)
            logged = gather_metrics({name: torch.as_tensor(value, device=device) for name, value in diagnostics.items()})
            if not all(np.isfinite(value) for value in logged.values()):
                raise FloatingPointError('nonfinite score critic or gradient evidence')
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar('charts/SPS', global_step / (now - started), global_step)
            writer.add_scalar('charts/interval_SPS', (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f'timing/{phase}_s', timing['total_s'], global_step)
            timer.reset()
            writer.flush()
            print(f'stage={stage} rollout={iteration}/{iterations} step={global_step} metrics={json.dumps(logged)}', flush=True)
            interval_start, interval_step = now, global_step

        if not all(torch.equal(value, actor_start[name]) for name, value in agent.actor.state_dict().items()):
            raise RuntimeError('fixed actor changed')
        for name, model in critics.items():
            if not all(torch.equal(value, frozen[name][key]) for key, value in model.state_dict().items()):
                raise RuntimeError('critic changed during holdout')
        report = summarize_gate(rows, args.seed)
        rng = np.random.default_rng(args.seed + 61)
        bootstrap = rng.integers(len(rows), size=(4096, len(rows)))
        report['horizons'] = {}
        for h in ('1', '4', '16', '64'):
            reference = np.asarray([row['horizons'][h]['reference'] for row in rows])
            report['horizons'][h] = {name: compare_gradients(reference, np.asarray([row['horizons'][h][name] for row in rows]), rng, bootstrap,
                                                           np.asarray([row['episodes'] for row in rows]))
                                     for name in critics}
        report.update(args=vars(args), fresh_initialization=True, checkpoint_loaded=False,
                      actor_unchanged=True, critics_frozen_for_holdout=True, gae_used=False,
                      value_bootstrap_used=False, reward_decoder_used=False,
                      objective='E[sum_(t=0)^999 gamma^t raw_reward_t] / frozen_global_reward_std',
                      centering='frozen per-episode-age reward mean; exact policy-independent baseline for the fixed-length task',
                      transitions=global_step, holdout=rows)
        with open(f'{run_dir}/gate.json', 'w') as output:
            json.dump(report, output, indent=2, allow_nan=False)
        if args.save_model:
            torch.save({'actor': agent.actor.state_dict(), 'critics': {name: model.state_dict() for name, model in critics.items()},
                        'obs_norm': {name: getattr(obs_norm, name) for name in ('means', 'variances', 'counts', 'epsilon', 'clip')},
                        'reward_scale': reward_scale, 'reward_center': reward_center, 'args': vars(args)},
                       f'{run_dir}/{args.exp_name}.cleanrl_model')
        writer.add_scalar('gate/passed', int(report['passed']), global_step)
        print('GATE_RESULT=' + json.dumps({k: v for k, v in report.items() if k not in ('args', 'holdout')}), flush=True)
        print(f'gate report saved to {run_dir}/gate.json', flush=True)


if __name__ == '__main__':
    main()
