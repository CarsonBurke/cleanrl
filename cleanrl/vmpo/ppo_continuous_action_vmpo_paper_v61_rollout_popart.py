# V-MPO v61 — rollout-level PopArt statistics (arXiv:1809.04474, p6).
# Isolated change from v60: each trajectory contributes its mean n-step target
# once to the online first/second moments, rather than one flattened-batch EMA.
# Apply sequential EMA in env-index order; the second moment uses squared
# trajectory means, not mean squared timestep targets. Preserve raw values.
# This follows the cited PopArt recipe; V-MPO's private update cadence is unknown.
# Keep v60's solved shared eta, raw inputs, fixed target hold and architecture.
# Initial Gaussian std remains explicit: compare runs at the same chosen value.
# CUDA FP32 is deliberate: tiny covariance budgets and host/device policy
# agreement take precedence over mixed-precision throughput in this reference.

import copy
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import TruncationBootstrapCache, gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer

DUAL_FLOOR = 1e-8
STD_FLOOR = 1e-6
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    num_envs: int = 64
    num_steps: int = 39
    num_replicas: int = 8
    env_threads: int = 2
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_update_period: int = 100
    epsilon_eta: float = 0.01
    epsilon_alpha_mean: float = math.sqrt(0.005 * 0.01)
    epsilon_alpha_covariance: float = math.sqrt(5e-6 * 5e-5)
    initial_alpha_mean: float = 1.0
    initial_alpha_covariance: float = 1.0
    initial_std: float = 1.0
    popart_rate: float = 1e-4
    popart_std_min: float = 1e-2
    popart_std_max: float = 1e6
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    log_interval: int = 10
    eval_interval: int = 100
    eval_envs: int = 8
    save_model: bool = True
    batch_size: int = 0
    num_iterations: int = 0


def validate_args(args):
    counts = (args.num_envs, args.num_steps, args.num_replicas, args.env_threads,
              args.target_update_period, args.log_interval, args.eval_interval, args.eval_envs)
    if min(counts) <= 0:
        raise ValueError("environment, replica, rollout and interval counts must be positive")
    if args.num_envs % args.num_replicas:
        raise ValueError("replicas must partition complete environment trajectories")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size // args.num_replicas < 2:
        raise ValueError("each replica needs at least two state-action samples")
    positive = (args.learning_rate, args.epsilon_eta, args.epsilon_alpha_mean,
                args.epsilon_alpha_covariance, args.initial_alpha_mean,
                args.initial_alpha_covariance, args.popart_std_min, args.popart_std_max)
    if any(not math.isfinite(x) or x <= 0 for x in positive):
        raise ValueError("learning rate, budgets, duals and PopArt scales must be finite and positive")
    if not math.isfinite(args.initial_std) or args.initial_std <= STD_FLOOR:
        raise ValueError("initial_std must exceed the numerical std floor")
    if not 0 < args.gamma <= 1 or not 0 < args.popart_rate <= 1:
        raise ValueError("gamma and PopArt rate must be in (0, 1]")
    if args.popart_std_min > args.popart_std_max:
        raise ValueError("PopArt scale bounds are reversed")
    return args


def layer_init(layer, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, initial_std=1.0):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(256, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 2 * action_dim), gain=0.01),
        )
        with torch.no_grad():
            self.policy_head[-1].weight[action_dim:].zero_()
            self.policy_head[-1].bias[action_dim:].fill_(math.log(math.expm1(initial_std - STD_FLOOR)))
        self.value_mlp = nn.Sequential(layer_init(nn.Linear(256, 256)), nn.ReLU())
        self.value_head = layer_init(nn.Linear(256, 1), gain=0.0)
        self.register_buffer("popart_mean", torch.zeros(()))
        self.register_buffer("popart_sq_mean", torch.ones(()))
        self.register_buffer("popart_std", torch.ones(()))

    def actor_layers(self):
        """Flat view of the actual actor modules for the shared host mirror."""
        return nn.Sequential(*self.trunk, *self.policy_head)

    def policy(self, observations):
        mean, raw_std = self.policy_head(self.trunk(observations)).chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + STD_FLOOR

    def forward(self, observations):
        features = self.trunk(observations)
        mean, raw_std = self.policy_head(features).chunk(2, dim=-1)
        value = self.value_head(self.value_mlp(features)).squeeze(-1)
        return mean, F.softplus(raw_std) + STD_FLOOR, value

    def value(self, observations):
        normalized = self.value_head(self.value_mlp(self.trunk(observations))).squeeze(-1)
        return normalized * self.popart_std + self.popart_mean

    @torch.no_grad()
    def update_popart(self, returns, rate, std_min, std_max):
        old_mean, old_std = self.popart_mean.clone(), self.popart_std.clone()
        trajectory_means = returns.mean(dim=0)
        count = trajectory_means.shape[0]
        decay = 1.0 - rate
        weights = rate * decay ** torch.arange(count - 1, -1, -1, device=returns.device, dtype=returns.dtype)
        new_mean = decay ** count * old_mean + (weights * trajectory_means).sum()
        new_sq_mean = decay ** count * self.popart_sq_mean + (weights * trajectory_means.square()).sum()
        new_std = (new_sq_mean - new_mean.square()).clamp_min(0).sqrt().clamp(std_min, std_max)
        # Output preservation is essential: statistics updates must not change
        # the raw bootstrap function underneath already computed returns.
        self.value_head.weight.mul_(old_std / new_std)
        self.value_head.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)
        self.popart_mean.copy_(new_mean)
        self.popart_sq_mean.copy_(new_sq_mean)
        self.popart_std.copy_(new_std)
        return (returns - new_mean) / new_std


class GaussianSampler:
    """Reusable host Normal draws; score raw samples, execute clipped actions."""
    def __init__(self, num_envs, low, high, seed):
        self.low, self.high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        shape = (num_envs, self.low.size)
        self.std = np.empty(shape, dtype=np.float32)
        self.raw = np.empty(shape, dtype=np.float32)
        self.action = np.empty(shape, dtype=np.float32)

    def __call__(self, logits):
        mean = logits[:, :self.low.size]
        np.logaddexp(0.0, logits[:, self.low.size:], out=self.std)
        self.std += STD_FLOOR
        self.rng.standard_normal(size=self.raw.shape, dtype=np.float32, out=self.raw)
        np.multiply(self.raw, self.std, out=self.raw)
        np.add(self.raw, mean, out=self.raw)
        if not (np.isfinite(self.raw).all() and np.isfinite(self.std).all()):
            raise FloatingPointError("nonfinite Gaussian behavior distribution or sample")
        np.clip(self.raw, self.low, self.high, out=self.action)
        return self.raw, self.action, mean, self.std


def gaussian_log_prob(mean, std, actions):
    return (-0.5 * ((actions - mean) / std).square() - std.log() - 0.5 * LOG_TWO_PI).sum(-1)


def gaussian_kls(old_mean, old_std, mean, std):
    """Appendix C frozen-old-covariance mean metric, covariance metric, full KL."""
    delta = mean - old_mean
    log_ratio = std.log() - old_std.log()
    covariance = (log_ratio + 0.5 * torch.expm1(-2 * log_ratio)).sum(-1)
    mean_kl = 0.5 * (delta / old_std).square().sum(-1)
    full_kl = covariance + 0.5 * (delta / std).square().sum(-1)
    return mean_kl, covariance, full_kl


def replica_samples(values, num_steps, num_envs, num_replicas):
    """Replica-major samples, keeping each complete unroll on one replica."""
    trailing = values.shape[1:]
    return values.reshape(num_steps, num_replicas, num_envs // num_replicas, *trailing).transpose(0, 1).reshape(
        num_replicas, -1, *trailing
    )


@torch.no_grad()
def replica_estep(advantages, epsilon):
    """Minimize Eq.4 averaged over replicas, with one eta and local weights.

    dL/deta = epsilon - mean_r KL(w_r || uniform_selected_r).
    Independent per-replica roots would solve a different constrained problem.
    Geometric bisection resolves raw reward scales without an eta learning rate.
    """
    count = advantages.shape[1] // 2
    threshold = torch.topk(advantages, count, dim=1).values[:, -1:]
    selected = advantages >= threshold
    selected_count = selected.sum(1)
    log_count = selected_count.log()
    maximum = advantages.max(1, keepdim=True).values
    centered = advantages - maximum
    # This upper endpoint makes every selected logit span <= epsilon,
    # hence also bounds their mean KL. Flat elites have slack at eta_floor.
    log_low = advantages.new_full((), math.log(DUAL_FLOOR))
    log_high = ((maximum - threshold).amax() / epsilon).clamp_min(DUAL_FLOOR).log()
    for _ in range(40):
        log_mid = (log_low + log_high) * 0.5
        logits = torch.where(selected, centered / log_mid.exp(), -torch.inf)
        log_weights = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        weights = log_weights.exp()
        kl = (weights * (torch.where(selected, log_weights, 0.0) + log_count[:, None])).sum(1)
        violation = kl.mean() > epsilon
        log_low = torch.where(violation, log_mid, log_low)
        log_high = torch.where(violation, log_high, log_mid)
    eta = log_high.exp()
    logits = torch.where(selected, centered / eta, -torch.inf)
    log_normalizer = torch.logsumexp(logits, dim=1)
    log_weights = logits - log_normalizer[:, None]
    weights = log_weights.exp()
    temperature_loss = (maximum[:, 0] + eta * (epsilon + log_normalizer - log_count)).mean()
    kl = (weights * (torch.where(selected, log_weights, 0.0) + log_count[:, None])).sum(1)
    ess = weights.square().sum(1).reciprocal()
    return weights, temperature_loss, kl, ess, selected_count, eta


def vmpo_loss(agent, duals, observations, actions, old_mean, old_std, advantages, normalized_returns, args):
    mean, std, values = agent(observations)
    alpha_mean, alpha_covariance = duals.clamp_min(DUAL_FLOOR).unbind()
    shard_advantages = replica_samples(advantages.detach(), args.num_steps, args.num_envs, args.num_replicas)
    weights, eta_loss, estep_kl, ess, selected_count, eta = replica_estep(shard_advantages, args.epsilon_eta)
    log_probs = replica_samples(gaussian_log_prob(mean, std, actions), args.num_steps, args.num_envs, args.num_replicas)
    policy_loss = -(weights * log_probs).sum(1).mean()
    mean_kl, covariance_kl, full_kl = (metric.mean() for metric in gaussian_kls(old_mean, old_std, mean, std))
    mean_penalty = alpha_mean * (args.epsilon_alpha_mean - mean_kl.detach()) + alpha_mean.detach() * mean_kl
    covariance_penalty = (alpha_covariance * (args.epsilon_alpha_covariance - covariance_kl.detach())
                          + alpha_covariance.detach() * covariance_kl)
    value_loss = 0.5 * (values - normalized_returns).square().mean()
    loss = policy_loss + mean_penalty + covariance_penalty + value_loss
    error = (values.detach() - normalized_returns) * agent.popart_std
    raw_targets = normalized_returns * agent.popart_std + agent.popart_mean
    metrics = torch.stack((policy_loss.detach(), value_loss.detach(), eta_loss.detach(),
                           mean_kl.detach(), covariance_kl.detach(), full_kl.detach(),
                           estep_kl.mean(), (ess / selected_count).mean(),
                           (args.num_replicas ** 2 / weights.square().sum()).detach(),
                           error.square().mean().sqrt(),
                           1 - error.var(unbiased=False) / (raw_targets.var(unbiased=False) + 1e-8),
                           std.detach().mean(), std.detach().amin(), std.detach().amax(), eta))
    return loss, metrics


@torch.no_grad()
def evaluate_policy(envs, mirror, seed, horizon):
    """One complete deterministic episode per row; isolated from training RNG/state."""
    mirror.refresh()
    observations, _ = envs.reset(seed=seed)
    count = envs.num_envs
    finished = np.zeros(count, dtype=bool)
    returns = np.zeros(count, dtype=np.float64)
    lengths = np.zeros(count, dtype=np.int64)
    steps = 0
    action_dim = int(np.prod(envs.single_action_space.shape))
    action = np.empty((count, action_dim), dtype=np.float32)
    for _ in range(horizon):
        logits = mirror(np.asarray(observations, dtype=np.float32))
        mean = logits[:, :action_dim]
        if not np.isfinite(mean).all():
            raise FloatingPointError("nonfinite deterministic policy")
        np.clip(mean, envs.single_action_space.low, envs.single_action_space.high, out=action)
        observations, rewards, terms, truncs, _ = envs.step(action)
        active = ~finished
        returns[active] += rewards[active]
        lengths[active] += 1
        finished |= terms | truncs
        steps += count
        if finished.all():
            break
    if not finished.all() or not np.isfinite(returns).all():
        raise RuntimeError("deterministic evaluation failed to produce finite complete episodes")
    return float(returns.mean()), float(returns.std()), float(lengths.mean()), steps


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU learner fallback")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id)
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and one complete rollout")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path("runs") / run_name
    with ExitStack() as resources:
        writer = SummaryWriter(str(run_dir))
        resources.callback(writer.close)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{name}|{value}|" for name, value in asdict(args).items()))
        writer.add_text("paper_reference", "arXiv:1909.12238, Eqs 1-6/25-27, Appendix F/Table 7. "
                        f"Raw observations/rewards; n-step (lambda=1); solved shared eta; {args.num_replicas} env-axis replicas; "
                        "no entropy, importance weighting, gradient clipping or annealing. "
                        "Unspecified Gym details chosen explicitly: shared ReLU 512/256+256 heads, "
                        f"{args.num_envs} envs, initial std={args.initial_std}, orthogonal mean-head initialization. "
                        "PopArt follows arXiv:1809.04474 p6: mean target per trajectory, sequential EMA in env-index order, "
                        "second moment of trajectory means; initial moments (0,1), before fitting. "
                        "V-MPO's private PopArt update cadence is not established. "
                        "Retains v60 numerical shared E-step dual minimization instead of one eta Adam step. "
                        "Runtime departures: Gymnasium-v4, phase warmup, host FP32 actor, CUDA FP32 learner. "
                        "Evaluation transitions logged separately and excluded from training step count.")
        envs = make_mujoco_vector_env(args.env_id, args.num_envs, num_threads=args.env_threads)
        resources.callback(envs.close)
        eval_envs = make_mujoco_vector_env(args.env_id, args.eval_envs, num_threads=args.env_threads)
        resources.callback(eval_envs.close)
        obs_shape = envs.single_observation_space.shape
        action_dim = int(np.prod(envs.single_action_space.shape))
        agent = Agent(int(np.prod(obs_shape)), action_dim, args.initial_std).to(device)
        target_agent = copy.deepcopy(agent).requires_grad_(False)
        duals = nn.Parameter(torch.tensor([args.initial_alpha_mean,
                                         args.initial_alpha_covariance], device=device))
        optimizer = optim.Adam([*agent.parameters(), duals], lr=args.learning_rate,
                               betas=(0.9, 0.999), eps=1e-8, fused=True)
        host_actor = make_host_mirror(target_agent.actor_layers(), args.num_envs)
        target_eval_actor = make_host_mirror(target_agent.actor_layers(), args.eval_envs)
        online_eval_actor = make_host_mirror(agent.actor_layers(), args.eval_envs)
        sampler = GaussianSampler(args.num_envs, envs.single_action_space.low,
                                  envs.single_action_space.high, args.seed)
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   fields={"observations": obs_shape, "actions": (action_dim,),
                                           "behavior_mean": (action_dim,), "behavior_std": (action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        nstep_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)

        def rollout_statistics(observations, actions, mean, std):
            # The stored host distribution is authoritative; never score a
            # clipped action or replace behavior statistics by rounded GPU ones.
            return agent.value(observations), gaussian_log_prob(mean, std, actions)

        def loss_fn(observations, actions, mean, std, advantages, targets):
            return vmpo_loss(agent, duals, observations, actions, mean, std, advantages, targets, args)

        def post_update(observations, actions, old_mean, old_std, old_log_probs):
            mean, std = agent.policy(observations)
            target_mean, target_std = target_agent.policy(observations)
            kls = gaussian_kls(old_mean, old_std, mean, std)
            log_ratio = gaussian_log_prob(mean, std, actions) - old_log_probs
            return torch.stack((*[kl.mean() for kl in kls],
                                (target_mean - old_mean).abs().amax(),
                                ((target_std - old_std) / old_std).abs().amax(),
                                log_ratio.mean(), log_ratio.square().mean().sqrt()))

        value_fn = agent.value
        popart_fn = agent.update_popart
        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            value_fn = graph_compile(value_fn)
            popart_fn = graph_compile(popart_fn)
            loss_fn = torch.compile(loss_fn, mode=args.compile_mode, fullgraph=True, dynamic=False)
            post_update = graph_compile(post_update)
        host_actor.refresh()
        start_time = time.perf_counter()
        phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
        writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
        warm = run_phase_warmup(envs, obs_norm=None, rew_norm=None,
                                act_fn=lambda obs: sampler(host_actor(obs))[1], horizon=horizon,
                                phase_offsets=phases, seed=args.seed)
        next_obs, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        timer = PhaseTimer()
        interval_start, interval_step = time.perf_counter(), global_step
        eval_steps = 0
        metric_names = ("losses/policy_loss", "losses/value_loss", "losses/temperature_loss",
                        "vmpo/mean_kl", "vmpo/covariance_kl", "vmpo/full_gaussian_kl",
                        "vmpo/e_step_kl", "vmpo/weight_ess_fraction", "vmpo/weight_ess",
                        "debug/value_rmse", "debug/value_explained_variance",
                        "debug/policy_std_mean", "debug/policy_std_min", "debug/policy_std_max", "vmpo/eta")
        post_names = ("vmpo/post_mean_kl", "vmpo/post_covariance_kl", "vmpo/post_full_gaussian_kl",
                      "debug/host_mean_max_error", "debug/host_std_max_relative_error",
                      "vmpo/post_log_ratio_mean", "vmpo/post_log_ratio_rms")
        for iteration in range(1, args.num_iterations + 1):
            bootstraps.reset()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    observations = next_obs
                    raw, action, mean, std = sampler(host_actor(observations))
                with timer.span("env", use_cuda=False):
                    new_obs, rewards, terms, truncs, infos = envs.step(action)
                with timer.span("transfer", use_cuda=False):
                    # All reusable host buffers are staged before either the
                    # mirror or sampler is invoked again.
                    transfer.push(step, rewards, terms, truncs, observations=observations,
                                  actions=raw, behavior_mean=mean, behavior_std=std)
                    bootstraps.push(step, truncs, infos)
                    next_obs = np.asarray(new_obs, dtype=np.float32)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        if not math.isfinite(episode_return):
                            raise FloatingPointError("nonfinite training episode return")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)
            with timer.span("returns"), torch.no_grad():
                batch = transfer.upload()
                observations = batch.fields["observations"].flatten(0, 1)
                actions = batch.fields["actions"].flatten(0, 1)
                old_mean = batch.fields["behavior_mean"].flatten(0, 1)
                old_std = batch.fields["behavior_std"].flatten(0, 1)
                raw_values, old_log_probs = rollout_statistics(observations, actions, old_mean, old_std)
                tail_values = value_fn(transfer.observation(next_obs))
                truncation_values = bootstraps.resolve(value_fn, device, batch_size=args.num_envs)
                advantages, returns = nstep_fn(batch.rewards, raw_values.view(args.num_steps, args.num_envs),
                                               batch.terminations, batch.truncations, truncation_values,
                                               tail_values, args.gamma, 1.0)
                advantages = advantages.flatten().clone()
                returns = returns.flatten().clone()
                normalized_returns = popart_fn(returns.view(args.num_steps, args.num_envs), args.popart_rate,
                                                args.popart_std_min, args.popart_std_max).flatten()
            promoted = iteration % args.target_update_period == 0
            should_log = iteration == 1 or iteration % args.log_interval == 0 or promoted or iteration == args.num_iterations
            with timer.span("update"):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                loss, metrics = loss_fn(observations, actions, old_mean, old_std, advantages, normalized_returns)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    duals.clamp_(min=DUAL_FLOOR)
            if should_log:
                with timer.span("diagnostics"), torch.no_grad():
                    post = post_update(observations, actions, old_mean, old_std, old_log_probs)
                    logged = gather_metrics({**dict(zip(metric_names, metrics)), **dict(zip(post_names, post)),
                                             "vmpo/alpha_mean": duals[0],
                                             "vmpo/alpha_covariance": duals[1],
                                             "popart/mean": agent.popart_mean, "popart/std": agent.popart_std,
                                             "debug/advantage_mean": advantages.mean(),
                                             "debug/advantage_std": advantages.std(unbiased=False)})
                if any(not math.isfinite(value) for value in logged.values()):
                    raise FloatingPointError(f"nonfinite learner metrics at step {global_step}: {logged}")
                logged["vmpo/post_mean_kl_ratio"] = logged["vmpo/post_mean_kl"] / args.epsilon_alpha_mean
                logged["vmpo/post_covariance_kl_ratio"] = logged["vmpo/post_covariance_kl"] / args.epsilon_alpha_covariance
                logged["vmpo/target_age_batches"] = (iteration - 1) % args.target_update_period
                logged["vmpo/target_promoted"] = float(promoted)
                logged["vmpo/learner_updates"] = iteration
                for name, value in logged.items():
                    writer.add_scalar(name, value, global_step)
            # This is deliberately not a KL acceptance gate. The published
            # algorithm deploys every T_target updates, even if a dual lags.
            if promoted:
                target_agent.load_state_dict(agent.state_dict())
                host_actor.refresh()
            if iteration % args.eval_interval == 0 or iteration == args.num_iterations:
                with timer.span("evaluation", use_cuda=False):
                    target_result = evaluate_policy(eval_envs, target_eval_actor, args.seed + 10000, horizon)
                    eval_steps += target_result[3]
                    online_result = target_result if promoted else evaluate_policy(
                        eval_envs, online_eval_actor, args.seed + 10000, horizon)
                    if not promoted:
                        eval_steps += online_result[3]
                writer.add_scalar("charts/deterministic_return", target_result[0], global_step)
                writer.add_scalar("eval/target_return_std", target_result[1], global_step)
                writer.add_scalar("eval/target_episode_length", target_result[2], global_step)
                writer.add_scalar("eval/online_deterministic_return", online_result[0], global_step)
                writer.add_scalar("eval/transitions", eval_steps, global_step)
                print(f"step={global_step} target_eval={target_result[0]:.1f} online_eval={online_result[0]:.1f}", flush=True)
                if args.save_model:
                    torch.save({"args": asdict(args), "agent": agent.state_dict(),
                                "target_agent": target_agent.state_dict(), "duals": duals.detach(),
                                "optimizer": optimizer.state_dict(), "global_step": global_step,
                                "learner_updates": iteration, "eval_transitions": eval_steps}, run_dir / "checkpoint.pt")
            if should_log:
                now = time.perf_counter()
                writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
                writer.add_scalar("charts/SPS", global_step / (now - start_time), global_step)
                writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
                for phase, timing in timer.summary().items():
                    writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
                timer.reset()
                interval_start, interval_step = time.perf_counter(), global_step
                writer.flush()
        print(f"completed run={run_name} training_steps={global_step} evaluation_steps={eval_steps}", flush=True)


if __name__ == "__main__":
    main()
