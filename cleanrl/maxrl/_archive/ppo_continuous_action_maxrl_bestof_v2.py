# MaxRL-bestof v2: the best-of-T weight with the EMPIRICAL CDF, dropping the Gaussian plug-in.
#
# v1 took J_T = E[max of T returns] exactly, giving a weight whose slope is T F(Z)^(T-1) --
# smooth in the sample's percentile, where tail_v1's hinge was flat below a quantile.  That
# fixed the hinge: at matched order the smooth weight beat the hard one by a wide margin
# (T=4 vs alpha=0.75 at 1M: 3611 against 2341), and the arm that drops the per-state scale
# led the PPO control at 2M (5959 against 5470), the first arm in this line to do so.
#
# But v1 read the percentile through a Gaussian: F(Z) = Phi(A/sigma(s)).  Its own GO/NO-GO
# said that is wrong.  The logged kurtosis of A/sigma ran 12 to 150 against the 3 a Gaussian
# would give, so the advantage distribution here is violently heavy tailed and Phi is not
# the percentile it claims to be.  The damage is concrete: extreme samples are read as
# percentile 1 and collect the full slope T, which is why weight_std_ratio reached 8.9 for
# T=2 and 57 for the lower arm -- a variance price paid for a mis-specified model, not for
# the objective.
#
# The fix is to stop modelling it.  F is only ever needed AT THE SAMPLES, so the empirical
# CDF of the batch serves, and the integral has a closed form on the order statistics:
#     h(z_(i)) = T * int_{z_(1)}^{z_(i)} F(z)^(T-1) dz,   F(z_(i)) = (i + 1/2) / N
# by cumulative trapezoid along the sorted sample.  No distribution is assumed, no scale
# head is needed, and at T=1 the integrand is one so h = z - min(z): PPO is still exactly
# the T=1 member, up to the affine shift that standardisation removes.
#
# Verified, not assumed.  On normals this reproduces v1's T-discriminating check exactly
# (d/dsigma E[max of T] = E[max of T standard normals]; T=2,4,8 give 0.5656/0.5636,
# 1.0304/1.0284, 1.4256/1.4244).  On Student-t(3), where Phi is badly wrong, the weight
# std ratio stays at 1.19/1.91/3.45 for T=2/4/8 instead of the 8.9-57 the Gaussian plug-in
# produced on the real advantages.  That variance is the whole of what v1 was paying.
#
# What the critic keeps.  Only the baseline b(s) = E[h(A) | s], regressed by squared error
# against a target fixed once per batch.  It is a state-dependent baseline and therefore
# free: it changes no expectation and only removes variance, exactly as V(s) does for A.
# The per-state SCALE is gone, and that is a result rather than a simplification -- v1's
# `unit` arms beat its `state` arms at every order, consistent with five earlier versions
# in which per-state reweighting was the part that cost.
#
# Hypothesis.  If v1's remaining gap to control was the mis-specified percentile, this
# closes it and T=2 beats PPO outright.  If the score tracks v1's `unit` arms instead, the
# Gaussian was never the binding constraint and the objective itself is what this domain
# tolerates but does not reward.
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
# Floor on the learned conditional scale, in units of the batch-standardised advantage
# (which has unit variance by construction, so this is an absolute and stable guard).
# It binds only where the head wants a scale near zero -- states with no advantage spread,
# where A/sigma is then bounded and the weight is small: a safe direction to be wrong in.
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
    """Toggles advantage standardization; it acts on the best-of-T advantage, so the clip
    range means the same thing at every T and the sweep is not a disguised learning-rate sweep"""
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

    # Best-of-T objective: the RL <-> ML dial of Table 2, taken exactly rather than hinged.
    bestof_order: float = 4.0
    """T in E[max of T returns]; 1 is exactly PPO, and need not be an integer"""
    bestof_side: str = "upper"
    """upper maximises E[max of T] (optimism); lower maximises E[min of T], the falsification arm"""
    bestof_coef: float = 0.5
    """weight of the baseline regression that amortises E[h(A) | s]"""

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


def bestof_orientation(side):
    """+1 maximises E[max of T], -1 maximises E[min of T].

    The lower arm is the identical transform applied to -A with the result negated, so it
    is the same code, the same curvature and the same variance profile pointed the other
    way -- and it coincides with the upper arm exactly at T=1, where both are PPO.
    """
    return 1.0 if side == "upper" else -1.0


def rank_weight(x, order):
    """h(z) = T * int F(z)^(T-1) dz on the EMPIRICAL distribution of ``x``, exactly.

    The best-of-T gradient weight needs the sample's percentile and nothing else, so the
    batch's own order statistics supply it: F(z_(i)) = (i + 1/2)/N, and the integral becomes
    a cumulative trapezoid along the sorted values.  No distribution is assumed -- which is
    the point, since the advantages here carry kurtosis 12 to 150 and the Gaussian plug-in
    v1 used read their extremes as percentile 1 and handed them the full slope T.

    At order 1 the integrand is identically one and this returns z - min(z), an affine image
    of the advantage; the standardisation in the loss removes the shift, so PPO remains the
    T=1 member.  The accumulation runs in float64 because a 32k-term cumsum in float32 loses
    the small increments at the bottom of the sort, which are exactly the ones order T is
    meant to down-weight smoothly rather than erase.
    """
    values, positions = torch.sort(x)
    count = x.numel()
    cdf = (torch.arange(count, device=x.device, dtype=torch.float64) + 0.5) / count
    density = cdf.pow(order - 1.0)
    trapezoid = 0.5 * (density[:-1] + density[1:]) * values.diff().to(torch.float64)
    climb = torch.cat((torch.zeros(1, dtype=torch.float64, device=x.device),
                       trapezoid.cumsum(0))) * order
    weights = torch.empty_like(x)
    weights[positions] = climb.to(x.dtype)
    return weights


def bestof_advantages(oriented, baseline, order):
    """w = h(A) - b(s), the best-of-T analogue of A = R - V(s).

    ``b`` is the conditional mean of ``h``, regressed directly, which keeps w centred per
    state whatever the shape of the advantage distribution -- so a poor baseline fit costs
    variance but never injects the state-dependent offset that would order the arms by a
    nuisance rather than by the mechanism under test.
    """
    return rank_weight(oriented, order) - baseline


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
        # initialisation stream as the baseline, so `bestof_order=1` (which never
        # backpropagates into this head) stays bit-identical to plain PPO.
        self.spread = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_spread(self, x):
        """b(s) = E[h(A) | s], the state-dependent baseline of the rank-weighted score.

        A location on the real line, so it is unconstrained.  It is the only statistic this
        version amortises: subtracting any function of s changes no expectation and only
        removes variance, which is the same licence V(s) has against the raw return.
        """
        return self.spread(x).flatten()

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
             old_values, oriented, weighted_target, args):
    """Clipped PPO surrogate on the BEST-OF-T advantage, plus the two head regressions.

    ``advantages`` already carries the rank transform.  ``weighted_target`` is h(A),
    evaluated once per batch: it is a function of the batch's order statistics alone, so it
    is constant in the parameters and is the baseline head's stationary regression target.
    ``oriented`` is retained for signature parity with the diagnostics and is unused here.
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
    # Gated at trace time: at T=1 the head is absent from the graph, gets no gradient,
    # and the update is bit-identical to plain PPO.
    if args.bestof_order <= 1.0:
        head_loss = torch.zeros((), device=loss.device)
    else:
        baseline = agent.get_spread(observations)
        head_loss = ((baseline - weighted_target) ** 2).mean()
        loss = loss + args.bestof_coef * head_loss
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy.mean().detach(),
                           old_approx_kl, approx_kl, clipfrac, head_loss.detach()))
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
    if not 1.0 <= args.bestof_order:
        raise ValueError("bestof_order must be at least 1; T=1 is PPO and T<1 is not an order")
    if args.bestof_side not in {"upper", "lower"}:
        raise ValueError(f"unknown bestof_side {args.bestof_side!r}")
    if args.bestof_coef < 0.0:
        raise ValueError("bestof_coef must be non-negative")
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
        writer.add_text("objective", "best-of-T: w = h(A) - b(s), h' = T Fhat^(T-1) on the batch ECDF; T=1 is PPO")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        # Clipped as two disjoint sets, because they are two different estimators sharing
        # one backward pass.  `spread` is a nuisance regression with no parameters in
        # common with the policy; under a single global clip its gradient is pure extra
        # norm mass, so the policy's effective step falls as the head's loss grows.  That
        # made the order sweep a learning-rate sweep: |g_core| is flat at 0.51 across
        # T=1/2/4/8 while |g_head| runs 0.0/1.81/1.28/1.08, dropping the post-clip policy
        # step to 0.500/0.137/0.185/0.214 -- a 3.7x handicap on T=2 that has nothing to do
        # with the objective, and that biases every arm against the T=1 control.  Split
        # this way the policy path sees exactly max_grad_norm, identical to the baseline.
        policy_parameters = list(agent.actor.parameters()) + list(agent.critic.parameters())
        head_parameters = list(agent.spread.parameters())
        assert len(policy_parameters) + len(head_parameters) == len(list(agent.parameters()))
        value_model = agent.get_value
        sign = bestof_orientation(args.bestof_side)

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values,
                       oriented, weighted_target):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns,
                            old_values, oriented, weighted_target, args)

        spread_model = agent.get_spread

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            spread_model = torch.compile(spread_model, fullgraph=True, dynamic=True,
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
                if args.bestof_order <= 1.0:
                    # The T=1 member in closed form: psi_1 is the identity, so w = A with
                    # no standardisation and no head, which is plain PPO to the last bit.
                    b_oriented = b_advantages
                    b_weights = b_advantages
                    b_weighted_target = b_advantages
                    baseline = torch.zeros_like(b_advantages)
                else:
                    # Standardised over the whole batch so the head's targets carry fixed
                    # units while the reward normaliser's scale drifts.  Minibatch
                    # standardisation in the loss is invariant to this global affine map,
                    # so it changes nothing the policy sees beyond fixing the head's units.
                    standardised = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                    b_oriented = sign * standardised
                    # Read before the update and trained after it, so the statistics a
                    # state is judged against never see the action actually taken there.
                    baseline = spread_model(b_obs)
                    b_weighted_target = rank_weight(b_oriented, args.bestof_order)
                    b_weights = sign * (b_weighted_target - baseline)
                # Descriptive only now, where v1 used it as a GO/NO-GO: nothing here assumes
                # a shape, so a kurtosis of 30 is a fact about the advantages rather than a
                # model violation.  Kept because it is the number that condemned the
                # Gaussian plug-in, and a drop toward 3 would mean this version is no
                # longer distinguishable from it.
                centred_o = b_oriented - b_oriented.mean()
                kurtosis = (centred_o ** 4).mean() / (centred_o ** 2).mean().clamp_min(1e-12) ** 2
                # How much of h(A)'s variance the state-dependent baseline explains. This is
                # the only thing the critic is asked for here, so a value near zero means the
                # head is dead weight and w is effectively unbaselined.
                target_residual = b_weighted_target - baseline
                baseline_ev = 1.0 - target_residual.var() / b_weighted_target.var().clamp_min(1e-8)
                # Held-out centring: w must have conditional mean zero for the baseline
                # argument to hold, and this is read before the head is trained on it.
                weight_mean = b_weights.mean() / b_weights.std().clamp_min(1e-12)
                # How far the transform moved from PPO, and what it cost in variance.
                centred_w = b_weights - b_weights.mean()
                centred_a = b_advantages - b_advantages.mean()
                weight_adv_rho = (centred_w * centred_a).mean() / (
                    centred_w.std(unbiased=False) * centred_a.std(unbiased=False) + 1e-12)
                weight_std_ratio = b_weights.std() / b_advantages.std().clamp_min(1e-12)
                # Realised dose: dh/dz in the top decile against dh/dz at the median.
                # Each consecutive pair of order statistics carries the slope at a KNOWN
                # percentile, T*F^(T-1), so averaging the per-pair slopes weights every
                # percentile equally and the ratio comes out at (1 - 0.9^T)/(0.55^T -
                # 0.45^T) whatever the advantages are shaped like -- 1.0 at T=1, 1.9 at
                # T=2, 6.8 at T=4.  (Dividing the two total rises instead would weight by
                # z-spacing and just report how stretched this batch's tail is.)  A
                # departure means the transform is not seeing the ranks it should.
                ranking = b_oriented.argsort()
                z_steps = b_oriented[ranking].diff()
                h_steps = b_weighted_target[ranking].diff()
                local_slope = h_steps / z_steps.clamp_min(1e-12)
                count = local_slope.numel()
                decile, middle = count // 10, count // 2
                top_slope = local_slope[-decile:].mean() / local_slope[
                    middle - decile // 2:middle + decile // 2].mean().clamp_min(1e-12)
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_weights[indices], b_returns[indices], b_values[indices],
                            b_oriented[indices], b_weighted_target[indices],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy_parameters, args.max_grad_norm)
                        nn.utils.clip_grad_norm_(head_parameters, args.max_grad_norm)
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
                "losses/head_loss": update_metrics[:updates, 6].mean(),
                "bestof/kurtosis": kurtosis,
                "bestof/baseline_ev": baseline_ev,
                "bestof/weight_mean": weight_mean,
                "bestof/weight_adv_rho": weight_adv_rho,
                "bestof/weight_std_ratio": weight_std_ratio,
                "bestof/top_slope": top_slope,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in ("losses/explained_variance", "bestof/weight_adv_rho",
                                   "bestof/weight_mean", "bestof/baseline_ev")):
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
