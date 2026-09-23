# MaxRL-tail v1: PPO's objective REPLACED by the superquantile, with PPO as its alpha=0 member.
#
# What MaxRL actually says.  The trajectory is a latent variable and only the outcome is
# graded, so maximum likelihood over latents gives Theorem 1: grad J = E[grad log pi |
# success] -- the score averaged over the posterior conditioned on a GOOD OUTCOME.  The
# 1/p of Table 2 is not the idea; it is that posterior's normaliser in a bandit.  The
# order-T row is the tell: w_T = (1-(1-p)^T)/p is the gradient of 1-(1-p)^T = pass@T, the
# probability that the BEST OF T attempts succeeds.  T=1 is RL, T->infinity is ML.
#
# Why v1-v6 were degenerate.  Appendix M.4 generalises to dense r by a LINEAR tilt,
# E[r grad log pi]/E[r].  In a single-task MDP that is exactly parallel to the ordinary
# policy gradient -- the normaliser is a scalar -- so there is nothing to reweight unless
# you invent a task distribution.  v1/v3/v4/v5 invented one by calling each state a task
# and paid a variance price to re-price states that the visitation measure already prices;
# v6 moved the same weight onto exploration allocation and tied with uniform.  All six
# lost to control.  The weight was never the content.
#
# The non-degenerate extrapolation.  Binary reward hides a fork: "condition on success"
# and "tilt linearly by r" coincide when r is Bernoulli and diverge when it is not.  The
# linear branch is the degenerate one.  The other branch -- condition on the UPPER TAIL of
# the return distribution -- reduces exactly to Theorem 1 for Bernoulli returns (the tail
# event IS success) and stays live for dense ones.  And it is pass@T verbatim: the action
# is the first token, the continuation is the rest of the chain, and best-of-T over
# continuations at level alpha = 1 - 1/T is what the tail expectation measures.
#
# The objective.  Maximise E_s[S_alpha(s)], the alpha-superquantile of the return, in
# place of E_s[V(s)].  Rockafellar-Uryasev gives the per-state form exactly:
#     u = q_alpha(s) + (Z - q_alpha(s))_+ / (1-alpha)        tail-transformed return
#     w = u - S_alpha(s)                                     tail advantage
# and as alpha -> 0 the hinge opens to the whole support, q cancels, and w -> Z - V(s) = A.
# PPO is the alpha=0 member of this family, not a control bolted alongside it.
#
# What the critic becomes.  V(s) is kept -- GAE needs it -- and is joined by a head
# emitting (q_alpha(s), E[(A-q)_+|s]/(1-alpha)) in advantage units, trained by its own
# proper losses (pinball for the quantile, squared error for the excess).  No EMA, no
# auxiliary reweighting of a correct gradient, nothing bolted on: the policy loss is the
# clipped PPO surrogate with w in place of A.
#
# Mechanism, stated so it can be wrong.  A signed advantage punishes below-average actions
# in proportion to how bad they were, which drives the policy onto the current mode.  The
# tail advantage is a hinge: everything outside the tail gets the SAME push, and only what
# could have been good is reinforced.  That is the RL-vs-ML distinction in one line, and on
# a dense-reward gait landscape it should keep mass on faster gaits the mean would abandon.
#
# Falsification.  `--tail-side lower` runs the identical transform on -A and negates the
# result: the lower superquantile, i.e. pessimism.  Same family, same code path, same
# variance profile, opposite direction, and it COINCIDES with upper at alpha=0.  If upper
# and lower move together, the effect is the hinge's variance and not its direction.
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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantage standardization; it acts on the tail advantage, so the clip range
    means the same thing at every alpha and the sweep is not a disguised learning-rate sweep"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy bonus"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 4
    """maximum physics threads; four balances rollout throughput with concurrent runs"""
    compile: bool = True
    """compile deterministic policy statistics, PPO loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # Superquantile objective: the RL <-> ML dial of Table 2, read as pass@T.
    tail_alpha: float = 0.5
    """tail level; 0 is exactly PPO (order T=1), and alpha = 1 - 1/T for best-of-T"""
    tail_side: str = "upper"
    """upper maximises the upper superquantile (optimism); lower is the falsification arm"""
    tail_coef: float = 0.5
    """weight of the quantile and excess regressions that amortise the tail statistics"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Public evaluation and historical GAE compatibility helpers; not the training path.
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def bootstrap_observations(next_obs, truncations, infos):
    """Replace autoreset observations with final observations at time limits."""
    bootstrap_obs = np.array(next_obs, copy=True)
    truncations = np.asarray(truncations, dtype=bool)
    if not np.any(truncations):
        return bootstrap_obs

    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")

    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        final_observation = final_observations[env_idx]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observation
    return bootstrap_obs


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    rollout_tail_value,
    gamma,
    gae_lambda,
):
    """Compute GAE with distinct bootstrap and reset-boundary semantics."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for t in reversed(range(rewards.shape[0])):
        ordinary_next_value = rollout_tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = torch.where(
            truncations[t].bool(),
            truncation_bootstrap_values[t],
            ordinary_next_value,
        )
        bootstrap_nonterminal = 1.0 - terminations[t]
        trace_nonterminal = 1.0 - torch.maximum(terminations[t], truncations[t])
        delta = rewards[t] + gamma * bootstrap_nonterminal * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def tail_orientation(side):
    """+1 maximises the upper superquantile, -1 the lower one.

    The lower arm is the identical transform applied to -A with the result negated, so it
    is the same code, the same hinge and the same variance profile pointed the other way --
    and it coincides with the upper arm exactly at alpha=0, where both are PPO.
    """
    return 1.0 if side == "upper" else -1.0


def tail_advantages(oriented, quantile, excess, alpha):
    """Rockafellar-Uryasev tail advantage: the superquantile's analogue of A = R - V(s).

    ``u = q + (A - q)_+/(1-alpha)`` is the RU integrand whose state-conditional mean is the
    superquantile, and ``excess`` estimates ``E[u|s] - q``; the additive ``q`` cancels
    between the two, leaving the hinge position as the head's only influence.

    As alpha -> 0 the hinge opens below the whole support: ``(A-q)_+ = A-q`` and
    ``E[A-q|s] = -q`` because ``E[A|s] = 0``, so ``w -> A`` whatever ``q`` is.  That limit
    is why PPO is a member of this family rather than an external control.
    """
    return (oriented - quantile).clamp_min(0.0) / (1.0 - alpha) - excess


def pinball_loss(residual, alpha):
    """Check loss on ``residual = target - prediction``, minimised at the alpha-quantile.

    Its subgradient is ``-alpha`` above the prediction and ``1-alpha`` below, so the
    stationary point sits where the mass below equals alpha -- which is what the logged
    ``tail/below_frac`` is checking against, and the only claim the head has to make.
    """
    return torch.maximum(alpha * residual, (alpha - 1.0) * residual)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
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
        # Constructed last on purpose: every preceding layer then draws the same
        # initialisation stream as the baseline, so `tail_alpha=0` (which never
        # backpropagates into this head) stays bit-identical to plain PPO.
        self.tail = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=1.0),
        )

    def get_tail(self, x):
        """(quantile, excess) of the oriented advantage at s, in advantage units.

        The excess is the conditional mean of a non-negative quantity and is therefore
        softplus-constrained; the quantile is a location on the real line and is not.
        """
        head = self.tail(x)
        return head[..., 0], F.softplus(head[..., 1])

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


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns,
             old_values, oriented, args):
    """Clipped PPO surrogate on the TAIL advantage, plus the two head regressions.

    ``advantages`` already carries the tail transform; ``oriented`` is the raw signed
    advantage the head regresses against, and the two are the same tensor at alpha=0.
    """
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
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
    # Gated at trace time: at alpha=0 the head is absent from the graph, gets no gradient,
    # and the update is bit-identical to plain PPO.
    if args.tail_alpha <= 0.0:
        tail_loss = torch.zeros((), device=loss.device)
    else:
        quantile, excess = agent.get_tail(observations)
        # Detached: the excess regression estimates E[u|s] GIVEN the hinge, and must not
        # drag the hinge toward whatever position makes its own target easy to fit.
        excess_target = (oriented - quantile.detach()).clamp_min(0.0) / (1.0 - args.tail_alpha)
        tail_loss = (pinball_loss(oriented - quantile, args.tail_alpha)
                     + (excess - excess_target) ** 2).mean()
        loss = loss + args.tail_coef * tail_loss
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy.mean().detach(),
                           old_approx_kl, approx_kl, clipfrac, tail_loss.detach()))
    return loss, metrics


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if not 0.0 <= args.tail_alpha < 1.0:
        raise ValueError("tail_alpha must lie in [0, 1); 0 is PPO and 1 is an empty tail")
    if args.tail_side not in {"upper", "lower"}:
        raise ValueError(f"unknown tail_side {args.tail_side!r}")
    if args.tail_coef < 0.0:
        raise ValueError("tail_coef must be non-negative")
    if args.ent_coef < 0.0:
        raise ValueError("ent_coef must be non-negative")
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
        writer.add_text("objective", "superquantile: w = (A-q)_+/(1-alpha) - E[(A-q)_+|s]/(1-alpha); alpha=0 is PPO")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value
        sign = tail_orientation(args.tail_side)

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values, oriented):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns,
                            old_values, oriented, args)

        tail_model = agent.get_tail

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            tail_model = torch.compile(tail_model, fullgraph=True, dynamic=True,
                                       options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
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
        update_metrics = torch.empty((max_updates, 7), device=device)
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
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate
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
                b_oriented = sign * b_advantages
                if args.tail_alpha <= 0.0:
                    # The alpha -> 0 limit in closed form; the head is dead code here.
                    b_weights = b_advantages
                    quantile = torch.zeros_like(b_advantages)
                    excess = torch.zeros_like(b_advantages)
                else:
                    # Read before the update and trained after it, so the tail a state is
                    # judged against never sees the realisation of the action taken there.
                    quantile, excess = tail_model(b_obs)
                    b_weights = sign * tail_advantages(b_oriented, quantile, excess, args.tail_alpha)
                # Held-out calibration. The head is read here, BEFORE it is trained on this
                # batch, so this is fresh data: the pinball minimiser puts exactly alpha of
                # the mass below the prediction, and a below_frac that drifts off alpha means
                # the head has memorised past advantage draws rather than learned their
                # conditional quantile -- the one failure mode whose training loss looks fine.
                below_frac = (b_oriented < quantile).float().mean()
                # Is the tail STATE-DEPENDENT, or has the head collapsed to a batch-wide
                # constant?  A constant quantile makes this a batch hinge, not a per-state
                # superquantile, and is the GO/NO-GO for the whole construction.
                quantile_dispersion = quantile.std() / b_oriented.std().clamp_min(1e-12)
                # Expect this to be SMALL and positive: the target carries the full
                # realisation noise of A, of which only a little is state-explainable, so it
                # reads like the critic's explained variance on returns, not like an R^2.
                # Its sign is the claim; its level is not.
                excess_target = (b_oriented - quantile).clamp_min(0.0) / (1.0 - max(args.tail_alpha, 1e-6))
                excess_ev = 1.0 - (excess_target - excess).var() / excess_target.var().clamp_min(1e-8)
                # How far the transform actually moved from PPO. rho of 1 means the tail
                # advantage is a positive affine image of A and nothing has changed.
                centred_w = b_weights - b_weights.mean()
                centred_a = b_advantages - b_advantages.mean()
                weight_adv_rho = (centred_w * centred_a).mean() / (
                    centred_w.std(unbiased=False) * centred_a.std(unbiased=False) + 1e-12)
                # The variance price, before standardisation washes the scale out.
                weight_std_ratio = b_weights.std() / b_advantages.std().clamp_min(1e-12)
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_weights[indices], b_returns[indices], b_values[indices],
                            b_oriented[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "losses/tail_loss": update_metrics[:updates, 6].mean(),
                "tail/below_frac": below_frac,
                "tail/quantile_dispersion": quantile_dispersion,
                "tail/excess_ev": excess_ev,
                "tail/excess_mean": excess.mean(),
                "tail/weight_adv_rho": weight_adv_rho,
                "tail/weight_std_ratio": weight_std_ratio,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in ("losses/explained_variance", "tail/excess_ev",
                                   "tail/weight_adv_rho")):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
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
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.ppo_eval import evaluate
            episodic_returns = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10,
                run_name=f"{run_name}-eval", Model=Agent, device=device, gamma=args.gamma,
            )
            for index, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, index)
            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
