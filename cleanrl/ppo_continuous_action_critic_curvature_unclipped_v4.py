# Critic-curvature ablation under the unclipped Adam control, v4.
# Isolates v3's measured directional critic curvature and current-gradient
# fallback with no gradient clipping or value clipping. The actor remains
# ordinary fused Adam with the baseline linear LR schedule and PPO surrogate.
# Critic moments advance once per minibatch; probe/candidate checks use frozen
# value targets and retain the best measured point, including the origin.
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
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))

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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = True
    """apply the baseline linear schedule to actor Adam only"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile deterministic policy statistics, PPO loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    log_action_scale: torch.Tensor

    def __init__(self, envs):
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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training retains native samples."""
        alpha, beta, value = self.get_policy_and_value(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


def place_trial(params, origins, directions, gradients, lengths, predictions):
    for slot, (group, saved, direction, raw) in enumerate(zip(params, origins, directions, gradients)):
        terms = []
        for parameter, origin, vector, gradient in zip(group, saved, direction, raw):
            parameter.copy_(origin - lengths[slot] * vector)
            terms.append(-(gradient * (parameter - origin)).sum())
        predictions[slot].copy_(torch.stack(terms).sum())


def prepare_trial(params, gradients, moments, seconds, steps, origins, directions,
                  lrs, probe_lrs, predictions, fallback, beta1, beta2, eps):
    probe_lrs.copy_(lrs)
    for slot, (group, raw, means, variances, counters, saved, vectors) in enumerate(
        zip(params, gradients, moments, seconds, steps, origins, directions)
    ):
        adam, current, products = [], [], []
        for parameter, gradient, mean, variance, step, origin in zip(group, raw, means, variances, counters, saved):
            origin.copy_(parameter)
            step.add_(1)
            mean.lerp_(gradient, 1.0 - beta1)
            variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            denominator = (variance / (1.0 - beta2 ** step)).sqrt() + eps
            vector = (mean / (1.0 - beta1 ** step)) / denominator
            adam.append(vector)
            current.append(gradient / denominator)
            products.append((gradient * vector).sum())
        uphill = torch.stack(products).sum() <= 0
        fallback[slot].copy_(uphill.float())
        for vector, historical, fresh in zip(vectors, adam, current):
            vector.copy_(torch.where(uphill, fresh, historical))
    place_trial(params, origins, directions, gradients, probe_lrs, predictions)


def propose_trial(params, origins, directions, gradients, before, after, probe_lrs,
                  prediction, candidate_lrs, candidate_prediction):
    actual = before - after
    residual = prediction - actual
    positive = prediction > 0
    finite = torch.isfinite(after) & torch.isfinite(prediction) & torch.isfinite(actual)
    curvature_positive = residual > 0
    optimum = probe_lrs * prediction / torch.where(curvature_positive, 2.0 * residual, torch.ones_like(residual))
    usable = curvature_positive & torch.isfinite(optimum) & (optimum > 0)
    proposal = torch.where(usable, optimum, 2.0 * probe_lrs)
    proposal = torch.where(finite, proposal, 0.5 * probe_lrs)
    candidate_lrs.copy_(torch.where(positive | ~finite, proposal, probe_lrs))
    place_trial(params, origins, directions, gradients, candidate_lrs, candidate_prediction)


def finish_trial(params, origins, directions, lrs, before, probe_loss, candidate_loss,
                 probe_lrs, candidate_lrs, probe_prediction, candidate_prediction, fallback):
    probe_good = torch.isfinite(probe_loss) & torch.isfinite(probe_prediction) & (probe_loss < before)
    incumbent = torch.where(probe_good, probe_loss, before)
    candidate_good = (torch.isfinite(candidate_loss) & torch.isfinite(candidate_prediction)
                      & torch.isfinite(candidate_lrs) & (candidate_loss < incumbent))
    selected = torch.where(candidate_good, 2, torch.where(probe_good, 1, 0))
    selected_length = torch.where(candidate_good, candidate_lrs, torch.where(probe_good, probe_lrs, 0.0))
    for slot, (group, saved, vectors) in enumerate(zip(params, origins, directions)):
        for parameter, origin, vector in zip(group, saved, vectors):
            # Select origin explicitly: zero times a nonfinite vector is not zero.
            chosen = origin - selected_length[slot] * vector
            parameter.copy_(torch.where(selected[slot] > 0, chosen, origin))
    valid_candidate = torch.isfinite(candidate_lrs) & (candidate_lrs > 0)
    smaller = torch.minimum(probe_lrs, torch.where(valid_candidate, candidate_lrs, probe_lrs))
    recovery = torch.where(probe_prediction > 0, 0.5 * smaller, probe_lrs)
    lrs.copy_(torch.where(selected > 0, selected_length, recovery))
    accepted_loss = torch.where(candidate_good, candidate_loss, incumbent)
    return torch.stack((probe_prediction, before - probe_loss, candidate_prediction,
                        before - candidate_loss, probe_lrs, candidate_lrs, lrs,
                        selected.float(), fallback, before - accepted_loss), dim=1)


class CriticCurvatureAdam(optim.Optimizer):
    """Critic-only trials with unclipped current gradients.

    Moments summarize observed gradients, so trials/rejection never advance or
    roll them back. The LR carries over, but curvature does not: next minibatch
    has a different objective/direction. Caller evaluates both losses in order.
    """

    def __init__(self, critic_params, lr=3e-4, betas=(0.9, 0.999), eps=1e-5, compile=True):
        groups = [list(critic_params)]
        flat = groups[0]
        if not flat or len({id(p) for p in flat}) != len(flat):
            raise ValueError("CriticCurvatureAdam requires nonempty distinct critic parameters")
        if not all(np.isfinite(v) and v > 0 for v in (lr, eps)):
            raise ValueError("LR and epsilon must be positive and finite")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Adam decays must lie in [0, 1)")
        device = flat[0].device
        if device.type != "cuda" or any(p.device != device for p in flat):
            raise ValueError("CriticCurvatureAdam requires parameters on one CUDA device")
        super().__init__([dict(params=group) for group in groups], dict(betas=betas, eps=eps))
        self.groups = groups
        self.lrs = torch.full((1,), lr, device=device)
        self.probe_lrs = torch.zeros_like(self.lrs)
        self.candidate_lrs = torch.zeros_like(self.lrs)
        self.probe_prediction = torch.zeros_like(self.lrs)
        self.candidate_prediction = torch.zeros_like(self.lrs)
        self.fallback = torch.zeros_like(self.lrs)
        self.origins = [[torch.empty_like(p) for p in group] for group in groups]
        self.directions = [[torch.empty_like(p) for p in group] for group in groups]
        self.moments, self.seconds, self.steps = [], [], []
        for group in groups:
            for p in group:
                self.state[p] = dict(step=torch.zeros((), device=device), exp_avg=torch.zeros_like(p),
                                     exp_avg_sq=torch.zeros_like(p))
            self.moments.append([self.state[p]["exp_avg"] for p in group])
            self.seconds.append([self.state[p]["exp_avg_sq"] for p in group])
            self.steps.append([self.state[p]["step"] for p in group])
        self._prepare = torch.compile(prepare_trial, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}) if compile else prepare_trial
        self._propose = torch.compile(propose_trial, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}) if compile else propose_trial
        self._finish = torch.compile(finish_trial, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}) if compile else finish_trial

    @torch.no_grad()
    def probe(self):
        gradients = []
        for group in self.groups:
            if any(p.grad is None or p.grad.is_sparse for p in group):
                raise RuntimeError("Each objective parameter needs a dense current gradient")
            gradients.append([p.grad for p in group])
        self.gradients = gradients
        settings = self.param_groups[0]
        beta1, beta2 = settings["betas"]
        self._prepare(self.groups, gradients, self.moments, self.seconds, self.steps,
                      self.origins, self.directions, self.lrs, self.probe_lrs,
                      self.probe_prediction, self.fallback, beta1, beta2, settings["eps"])
        return self.probe_prediction

    @torch.no_grad()
    def propose(self, before, probe_loss):
        self._propose(self.groups, self.origins, self.directions, self.gradients,
                      before, probe_loss, self.probe_lrs, self.probe_prediction,
                      self.candidate_lrs, self.candidate_prediction)
        return self.candidate_prediction

    @torch.no_grad()
    def finish(self, before, probe_loss, candidate_loss):
        return self._finish(self.groups, self.origins, self.directions, self.lrs,
                            before, probe_loss, candidate_loss, self.probe_lrs,
                            self.candidate_lrs, self.probe_prediction, self.candidate_prediction,
                            self.fallback)


def value_loss(newvalue, returns):
    return 0.5 * (newvalue.view(-1) - returns).square().mean()


def critic_objective(agent, observations, returns, args):
    """Same weighted value objective as PPO; no actor/distribution computation."""
    return (args.vf_coef * value_loss(agent.get_value(observations), returns)).reshape(1)


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, args):
    """Policy-clipped PPO and unclipped value MSE on native Beta samples."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    v_loss = value_loss(newvalue, returns)
    entropy_loss = entropy.mean()
    weighted_value_loss = v_loss * args.vf_coef
    loss = pg_loss - args.ent_coef * entropy_loss + weighted_value_loss
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    return loss, metrics, weighted_value_loss.reshape(1)


def curvature_metrics(checks):
    metrics = {}
    for slot, name in enumerate(("critic",)):
        values = checks[:, slot]
        prefix = f"curvature/{name}/"
        for column, label in ((0, "probe_prediction"), (1, "probe_actual"),
                              (2, "candidate_prediction"), (3, "candidate_actual")):
            measurements = values[:, column]
            finite = torch.isfinite(measurements)
            # Rejected nonfinite trials remain visible as an explicit rate;
            # averages describe finite observations, not a fabricated decrease.
            metrics[prefix + label + "_finite_mean"] = torch.where(finite, measurements, 0.0).sum() / finite.sum().clamp_min(1)
            metrics[prefix + label + "_nonfinite_fraction"] = (~finite).float().mean()
        metrics.update({
            prefix + "probe_lr_min": values[:, 4].min(),
            prefix + "probe_lr_max": values[:, 4].max(),
            prefix + "next_lr": values[-1, 6],
            prefix + "origin_fraction": (values[:, 7] == 0).float().mean(),
            prefix + "probe_fraction": (values[:, 7] == 1).float().mean(),
            prefix + "candidate_fraction": (values[:, 7] == 2).float().mean(),
            prefix + "fallback_fraction": values[:, 8].mean(),
            prefix + "accepted_decrease": values[:, 9].mean(),
        })
    return metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
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
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        optimizer = CriticCurvatureAdam(agent.critic.parameters(), lr=args.learning_rate, compile=args.compile)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, args)

        def check_model(observations, returns):
            return critic_objective(agent, observations, returns, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            check_model = torch.compile(check_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = make_host_mirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        prediction_metrics = torch.empty((max_updates, 1, 10), device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                actor_optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
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

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs = rollout_statistics(b_obs, b_native)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        minibatch = (
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices],
                        )
                        loss, metrics, objectives = loss_model(*minibatch)
                        before_loss = objectives.detach().clone()
                        update_metrics[updates].copy_(metrics)
                        actor_optimizer.zero_grad(set_to_none=True)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.probe()
                        actor_optimizer.step()
                        with torch.no_grad():
                            probe_loss = check_model(minibatch[0], minibatch[4]).clone()
                            optimizer.propose(before_loss, probe_loss)
                            candidate_loss = check_model(minibatch[0], minibatch[4])
                            prediction_metrics[updates].copy_(optimizer.finish(before_loss, probe_loss, candidate_loss))
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            checks = prediction_metrics[:updates]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                **curvature_metrics(checks),
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(
                {
                    "model": agent.state_dict(),
                    "args": vars(args),
                    "obs_norm": {
                        "means": torch.from_numpy(obs_norm.means.copy()),
                        "variances": torch.from_numpy(obs_norm.variances.copy()),
                        "counts": torch.from_numpy(obs_norm.counts.copy()),
                        "epsilon": obs_norm.epsilon,
                        "clip": obs_norm.clip,
                    },
                },
                model_path,
            )
            print(f"model saved to {model_path}")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
