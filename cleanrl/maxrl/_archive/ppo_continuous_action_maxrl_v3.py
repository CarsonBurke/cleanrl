# MaxRL-PPO v3: latent-branch maximum likelihood on the DENSE reward, critic kept.
#
# What v1/v2 got wrong, measured over 18 runs at 8M on HalfCheetah:
#   * v1 manufactured a Bernoulli by thresholding the return against a quantile bar.
#     That bar is self-referential, so phat sat pinned near 1-q forever (0.55 at q=0.8,
#     0.15 at q=0.9) and could never RISE the way a real competence signal does. The
#     paper's whole temporal story -- slow early, overtakes later -- was unavailable.
#   * Reweighting the GAE advantage by any w(s) lost: 6057 (MaxRL direction) and 7299
#     (its mirror) against a bit-identical PPO control at 7991.
#   * The critic-free ports collapsed to 586-1716, because one scalar per segment
#     cannot replace per-timestep GAE credit assignment.
#
# The part of the paper those runs never tested is the LATENT GENERATION model, which is
# where its actual content lives: p(y|x) = sum_z m(z|x) I{f(z)=y}. The outcome's
# probability is a SUM OVER LATENT ROUTES, and Theorem 1 gives the ML gradient as
# E[grad log m(z|x) | f(z)=y*] -- the score averaged over SUCCESSFUL rollouts. The paper
# is explicit that "the latent variable is the successful trajectory that produced it".
# So the latent IS the trajectory, and that expectation needs SEVERAL trajectories from
# one x. A critic cannot supply it: V(s) is a scalar summary of how good a state is and
# carries no representation of how many distinct routes leave it. Difficulty pricing and
# route multiplicity are ORTHOGONAL axes, and v1's error was treating them as rivals.
#
# v3 therefore uses both, on the axes they belong to:
#   * the CRITIC does within-trajectory credit assignment (GAE), which is what makes a
#     1000-step dense-reward MuJoCo task tractable at all;
#   * CLONED MuJoCo STATE does the latent marginalisation. Every segment_length steps,
#     group-mates are teleported onto their leader's (qpos, qvel), so N envs become N
#     i.i.d. continuations z_1..z_N of ONE x. This is exact, not an approximation.
#
# And nothing is binarised. The reward stays dense. The paper's Appendix M.4 extends
# MaxRL to non-binary rewards only as the crude ratio advantage (r - mu)/mu, shows it
# beating GRPO by a large margin on a continuous-reward maze -- with GRPO's Best@32
# COLLAPSING, the diversity collapse its weight function predicts -- and then leaves
# "the full theoretical treatment and broader empirical evaluation of non-binary rewards
# to future work". cleanrl/shared/continuous_maclaurin.py supplies that missing piece:
# the truncated failure series with the failure INDICATOR replaced by failure MASS
# q = 1 - r, which reduces element-for-element to the authors' released binary estimator
# (pinned in tests/test_continuous_maclaurin.py) and interpolates in between.
#
# Segment returns are mapped to [0,1] against GLOBAL EMA quantiles, never per-group ones:
# a per-group scale would force every group to the same spread and erase precisely the
# cross-group difficulty variation the ML weighting exists to exploit (branch_v1's bug).
#
# The ML term is ADDITIVE, not multiplicative -- v1 measured that reweighting the
# advantage loses 24% while adding a bounded outcome term is the one thing that did not
# (tail 8074 vs control 7991). norm_adv is OFF by default: under per-minibatch
# standardisation the advantage blend is renormalised away and each arm silently
# re-centres its own baseline, which confounded the v2 comparison.
#
# Hypothesis: per-timestep GAE fixes credit assignment WITHIN a branch, the continuous
# failure series fixes the outcome weighting ACROSS branches, and because the two act on
# different axes they should compose rather than cancel. The controls that make this
# falsifiable are --maxrl-clone False (no branching at all) and --maxrl-beta 0 (branching
# paid for, ML term off), which separate the cost of collapsing env diversity from the
# benefit of the ML correction.
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

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
from cleanrl.shared.continuous_maclaurin import continuous_maclaurin_weights
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
# log1p(-p) must stay finite and w_T(p)=-expm1(T*log1p(-p))/p must not divide by 0.
PASS_FLOOR, PASS_CEIL = 1e-4, 1.0 - 1e-6
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
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 250
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
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # MaxRL v3: latent-branch maximum likelihood on the dense reward.
    maxrl_estimator: Literal["continuous", "binary", "grpo"] = "continuous"
    """continuous is the dense failure series; binary thresholds r at 0.5 (the v1 mistake,
    kept as an ablation); grpo is (r-mean)/std, the sharpening weight the paper critiques"""
    maxrl_order: int = 8
    """truncation order T of the failure series; T=1 is REINFORCE, T>=N is exact ML"""
    maxrl_beta: float = 1.0
    """weight of the ML outcome advantage, in units of A_GAE's std; 0 disables the term
    while still paying for cloning, which is the control that prices branching alone"""
    maxrl_clone: bool = True
    """teleport group-mates onto their leader's state; False leaves envs independent and
    makes the run plain PPO, the control that prices everything else"""
    group_size: int = 8
    """branches per state: the N of Theorem 1's E[. | f(z)=y*]"""
    segment_length: int = 125
    """steps between re-cloning; also the horizon of the outcome each branch is scored on"""
    maxrl_reward_quantile: float = 0.05
    """segment returns map to [0,1] against the (q, 1-q) global EMA quantiles"""
    maxrl_tau_ema: float = 0.95
    """EMA retention for those quantiles; damps per-batch jitter without pinning the scale"""

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
        # v3 needs no auxiliary head: the outcome probability comes from real branch
        # returns, not from a learned Bernoulli. So this is the baseline's own critic,
        # constructed in the baseline's order and drawing its identical RNG stream.
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


def clone_group_states(bases, normalized_obs, group_size):
    """Teleport every group-mate onto its leader's state: N branches from ONE x.

    This is what makes Theorem 1's E[. | f(z) = y*] estimable at all. MujocoEnv.set_state
    writes qpos/qvel straight into the mjData the native pool steps in place, so the
    group-mates become bit-identical continuations rather than merely similar states.

    The leader (member 0) is never written to, so its trajectory remains a genuine
    uninterrupted on-policy episode and is the only one whose episodic return is
    reported. Members inherit the leader's NORMALIZED observation verbatim: they are in
    the identical physical state, so renormalising would only risk double-counting the
    running observation statistics.
    """
    for start_index in range(0, len(bases), group_size):
        leader = bases[start_index]
        qpos, qvel = leader.data.qpos.copy(), leader.data.qvel.copy()
        # qacc_warmstart is the solver's scratch guess carried over from the leader's
        # previous step. set_state does not touch it, and leaving each member with its
        # own stale guess makes the branches differ in the last bits (~5e-15 on
        # HalfCheetah). Statistically irrelevant, but inheriting it is free and makes a
        # branch an exact continuation rather than an almost-exact one.
        warmstart = leader.data.qacc_warmstart.copy()
        for member in range(start_index + 1, start_index + group_size):
            bases[member].set_state(qpos, qvel)
            bases[member].data.qacc_warmstart[:] = warmstart
            normalized_obs[member] = normalized_obs[start_index]


def normalize_segment_returns(segment_returns, low, high):
    """Map dense segment returns onto [0,1] against GLOBAL bounds.

    Deliberately NOT per-group: normalising inside a group would force every group to
    the same spread and destroy the cross-group difficulty variation that the failure
    series exists to exploit -- an easy state, whose branches all do well, must be able
    to read as r ~ 1 across the board while a hard one reads r ~ 0.
    """
    return ((segment_returns - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def group_ml_advantages(rewards, args):
    """Outcome advantage per branch, shaped (segments, groups, group_size).

    ``rewards`` are the [0,1]-normalised segment returns. The continuous estimator is
    the order-T failure series with failure mass in place of the failure indicator;
    scaling its weights by group_size puts them on the scale of r itself, so beta keeps
    meaning the same thing as the group size changes.
    """
    group_size = rewards.shape[-1]
    if args.maxrl_estimator == "grpo":
        centred = rewards - rewards.mean(dim=-1, keepdim=True)
        return centred / (rewards.std(dim=-1, keepdim=True, unbiased=False) + 1e-8)
    scored = (rewards > 0.5).to(rewards.dtype) if args.maxrl_estimator == "binary" else rewards
    weights = continuous_maclaurin_weights(scored, args.maxrl_order)
    return (weights * group_size).to(rewards.dtype)


def blend_advantages(advantages, outcome, args):
    """Add the ML outcome term to GAE, holding the total advantage scale fixed.

    v1 measured that MULTIPLYING the advantage by an outcome-derived weight costs 24%
    while ADDING a bounded outcome term costs nothing, so the ML signal enters additively.
    Matching the outcome term to A_GAE's std makes beta scale-free, and dividing by
    sqrt(1+beta^2) keeps the blended advantage at A_GAE's own scale -- which matters here
    precisely because norm_adv is off and PPO's effective step size now tracks that scale.
    """
    if args.maxrl_beta == 0.0:
        return advantages
    scaled = outcome * (advantages.std() / (outcome.std() + 1e-8))
    return (advantages + args.maxrl_beta * scaled) / math.sqrt(1.0 + args.maxrl_beta ** 2)


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling."""
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
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
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
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if args.maxrl_order < 1:
        raise ValueError("maxrl_order is a truncation level T >= 1 (T=1 is REINFORCE)")
    if not 0.0 < args.maxrl_reward_quantile < 0.5:
        raise ValueError("maxrl_reward_quantile must lie strictly inside (0, 0.5)")
    if not 0.0 <= args.maxrl_tau_ema < 1.0:
        raise ValueError("maxrl_tau_ema must lie in [0, 1)")
    if args.maxrl_beta < 0.0:
        raise ValueError("maxrl_beta must be non-negative")
    if args.group_size < 2:
        raise ValueError("a group needs at least two branches to marginalise over")
    if args.num_envs % args.group_size:
        raise ValueError("num_envs must be a whole number of groups")
    if args.num_steps % args.segment_length:
        raise ValueError("segment_length must tile the rollout")
    args.num_groups = args.num_envs // args.group_size
    args.segments_per_rollout = args.num_steps // args.segment_length
    if args.maxrl_clone:
        if args.env_backend not in {"auto", "native"}:
            raise ValueError("state cloning requires the native MuJoCo backend")
        horizon = episode_horizon(args.env_id)
        if horizon % args.segment_length or horizon % args.num_steps:
            raise ValueError(f"segment_length and num_steps must divide the {args.env_id} "
                             f"episode horizon ({horizon}) so clones never straddle a reset")
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
        writer.add_text("maxrl", f"estimator={args.maxrl_estimator}; T={args.maxrl_order}; "
                                 f"beta={args.maxrl_beta}; clone={args.maxrl_clone}; "
                                 f"group={args.group_size}; segment={args.segment_length}; "
                                 f"reward q={args.maxrl_reward_quantile} EMA {args.maxrl_tau_ema}; "
                                 f"norm_adv={args.norm_adv}")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values in one forward over the uploaded rollout."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns,
                            old_values, args)

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
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
        update_metrics = torch.empty((max_updates, 6), device=device)
        # Reward bounds are device-resident state: reading them would sync the optimizer path.
        reward_low = torch.zeros((), device=device)
        reward_high = torch.zeros((), device=device)
        bounds_initialised = False
        maxrl_diagnostics = None
        bases = None
        if args.maxrl_clone:
            bases = getattr(envs, "_bases", None)
            if bases is None or len(bases) != args.num_envs or not hasattr(bases[0], "set_state"):
                raise RuntimeError("state cloning needs the native MuJoCo backend's base envs")
        discounts = args.gamma ** torch.arange(args.segment_length, device=device, dtype=torch.float32)
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
                    if bases is not None and step % args.segment_length == 0:
                        # Re-form the groups: from here, group-mates are i.i.d. branches
                        # of one x until the next boundary.
                        clone_group_states(bases, next_obs_np, args.group_size)
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
                        # A clone's episode is stitched from teleports, so its return is
                        # not a real trajectory return. Only leaders are ever reported.
                        if bases is not None and index % args.group_size:
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
                b_returns = returns.flatten().clone()
                if args.maxrl_beta > 0.0:
                    # One discounted outcome per branch over its own segment, truncated at
                    # segment_length with no bootstrap: the branch's f(z), scored on the
                    # dense reward and never thresholded.
                    shaped = batch.rewards.view(args.segments_per_rollout, args.segment_length,
                                                args.num_groups, args.group_size)
                    segment_returns = (shaped * discounts[None, :, None, None]).sum(dim=1)
                    flat_returns = segment_returns.flatten()
                    batch_low = torch.quantile(flat_returns, args.maxrl_reward_quantile)
                    batch_high = torch.quantile(flat_returns, 1.0 - args.maxrl_reward_quantile)
                    if bounds_initialised:
                        reward_low.mul_(args.maxrl_tau_ema).add_(batch_low, alpha=1.0 - args.maxrl_tau_ema)
                        reward_high.mul_(args.maxrl_tau_ema).add_(batch_high, alpha=1.0 - args.maxrl_tau_ema)
                    else:
                        reward_low.copy_(batch_low)
                        reward_high.copy_(batch_high)
                        bounds_initialised = True
                    branch_rewards = normalize_segment_returns(segment_returns, reward_low, reward_high)
                    outcome = group_ml_advantages(branch_rewards, args)
                    # Every timestep of a branch inherits its branch's outcome.
                    outcome_flat = outcome.unsqueeze(1).expand(
                        -1, args.segment_length, -1, -1
                    ).reshape(args.num_steps, args.num_envs)
                    group_means = branch_rewards.mean(dim=-1)
                    maxrl_diagnostics = torch.stack((
                        reward_low, reward_high, branch_rewards.mean(), branch_rewards.std(),
                        group_means.std(), branch_rewards.std(dim=-1).mean(),
                        outcome.std(), outcome.abs().max(),
                    ))
                    advantages = blend_advantages(advantages, outcome_flat, args)
                b_advantages = advantages.flatten().clone()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
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
            metric_tensors = {
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
            }
            if maxrl_diagnostics is not None:
                # across_group_std is the go/no-go test: it is the spread of per-state
                # difficulty across branch groups. If it collapses toward zero there is
                # no cross-state difficulty signal for the failure series to weight and
                # the ML term can only add variance, whatever T or beta say.
                metric_tensors.update(zip((
                    "maxrl/reward_low", "maxrl/reward_high",
                    "maxrl/branch_reward_mean", "maxrl/branch_reward_std",
                    "maxrl/across_group_std", "maxrl/within_group_std",
                    "maxrl/outcome_std", "maxrl/outcome_absmax",
                ), maxrl_diagnostics.unbind()))
            logged = gather_metrics(metric_tensors)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
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
