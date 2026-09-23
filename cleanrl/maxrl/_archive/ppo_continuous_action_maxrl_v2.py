# MaxRL-PPO v2: is difficulty-reweighting the WRONG direction for an MDP?
#
# v1 measured MaxRL's weight w_T(p) = (1-(1-p)^T)/p, which upweights states where
# above-bar outcomes are RARE, on HalfCheetah. It lost monotonically in the dose:
#   off 7944 @7M, tail_b10 7807, weight_T8 6025, weight_T16_q0.9 5546.
# The pass head was NOT degenerate (phat_std ~ 0.28), so w_T genuinely varied; the
# mechanism ran and the mechanism is what hurt.
#
# Diagnosis -- a structural mismatch, not a hyperparameter.
# MaxRL's weight is well-posed because an LLM's prompt distribution is EXOGENOUS: you
# cannot decline a hard prompt, so spending gradient where you fail is the only way to
# ever pass it. An MDP's state distribution is ENDOGENOUS: the policy chooses where it
# goes. Upweighting low-p states optimises competence in a region the agent should be
# LEAVING, and it does so with a feedback loop the LLM setting has no analogue for.
#
# The paper (Table 2) treats GRPO's w = 1/sqrt(p(1-p)) as a pathology because it inverts
# -- upweighting p->1 -- and "sharpens the distribution". HYPOTHESIS: under an endogenous
# state distribution that sharpening is CORRECT, and MaxRL and PPO-with-GRPO-weights sit
# on the right and wrong sides of the same axis for control.
#
# Three arms isolate direction from variance. All share v1's phat head and mean-1
# normalisation, so effective policy LR is identical across them and against v1.
#   sharpen: w_T^s(p) = (1-p^T)/(1-p) = w_T(1-p), the EXACT mirror -- same bounded
#            family [1,T], same T, increasing in p instead of decreasing. One sign flip
#            against v1's weight arm; everything else is byte-identical.
#   shuffle: v1's own w_T(p), randomly permuted across the batch. Same multiplier
#            distribution, correlation with the state destroyed. Separates "the
#            direction is wrong" from "any per-state reweighting just adds variance".
#            If shuffle ~ off then direction is the story; if shuffle ~ weight, variance is.
#   sharpen + --maxrl-pass-detach False: the one genuine critic-side lever MaxRL offers.
#            BCE on the success event IS maximum likelihood on that event, so an attached
#            trunk lets the ML objective shape the shared value representation as an
#            auxiliary task. v1 kept it detached so gains were attributable to reweighting
#            alone; this asks whether the representation itself was the missing half.
#
# KNOWN DEFECTS in the 8M runs this file produced (red-team, post hoc). None of them
# change the headline -- that rests on returns against a control confirmed bit-identical
# -- but they bound what the arms can be read to prove:
#  1. norm_adv makes the mean-1 normalisation LITERALLY INERT: per-minibatch
#     standardisation is invariant to positive rescaling. Worse, corr(w,A) != 0 makes
#     mean(wA) systematically negative for `weight` (-0.65 std units) and positive for
#     `sharpen` (+0.61); norm_adv subtracts that, and a uniform advantage shift is NOT
#     symmetric under PPO's max(pg1, pg2) clip. So `sharpen` is NOT `weight` up to one
#     sign flip -- each arm re-centres its own baseline, and the weight-vs-sharpen gap
#     is confounded by clipping asymmetry, not purely a direction effect.
#  2. reallocation_rho correlates a STATE-ONLY multiplier against the STATE-CONDITIONAL
#     residual A, where E[A|s] ~ 0 for a fitted critic, so it is ~0 by construction and
#     cannot discriminate the arms. Measured 8M: shuffle -0.003, sharpen +0.035. The
#     axis that reads the mechanism out is corr(w, returns) or corr(w, V(s)).
#  3. `shuffle` draws its permutation from shuffle_generator, the same generator
#     device_minibatches uses, so that arm consumes 11 randperm draws per iteration
#     against every other arm's 10 -- a different minibatch partition stream for
#     precisely the arm meant to isolate one variable. Single seed, so this folds an
#     uncontrolled seed effect into the control.
# Fixes for any successor: separate generator seeded seed+1337; rank against returns;
# normalise once over the full batch (or --norm-adv False) so mean-1 is real.
#
# Note on the critic: MaxRL has no critic-side content of its own. Its group mean rbar
# and a value baseline are SUBSTITUTES -- the group exists precisely because an LLM has
# no V. So "apply MaxRL to the critic" is not a well-posed port; the attached-trunk arm
# above is the real and only lever, and the critic-free ports live in the _pure_ and
# _branch_ files where the group baseline legitimately replaces V.
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

    # MaxRL: maximum-likelihood reweighting of the policy gradient.
    maxrl_mode: Literal["off", "weight", "sharpen", "shuffle"] = "sharpen"
    """off is exact baseline PPO; weight is v1's MaxRL w_T(p); sharpen is its mirror w_T(1-p);
    shuffle permutes w_T(p) across the batch as the direction-vs-variance control"""
    maxrl_order: int = 8
    """truncation order T of the Maclaurin expansion; caps the weight at w_T(0)=T"""
    maxrl_quantile: float = 0.8
    """the success bar is this quantile of the batch's GAE return targets"""
    maxrl_tau_ema: float = 0.95
    """EMA retention for the success bar; damps per-batch quantile jitter"""
    maxrl_pass_coef: float = 0.5
    """coefficient of the pass head's BCE loss"""
    maxrl_pass_detach: bool = True
    """detached: the BCE loss cannot reshape the value function, so any gain is
    attributable to reweighting. False is the auxiliary-representation arm."""

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
        # The critic is split so a second head can read the same features. Module
        # construction order matches the baseline's two Sequentials, so every shared
        # parameter draws the identical RNG stream; pass_head is built last and is the
        # only new draw, keeping --maxrl-mode off bit-identical to ppo_continuous_action.
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * self.action_dim), std=0.01),
        )
        # Success-probability head: phat(s) estimates P(return target > bar | s).
        # Small init keeps phat near 1/2, hence w_T near its midpoint, before it learns.
        self.pass_head = layer_init(nn.Linear(64, 1), std=0.01)

    def get_value(self, x):
        return self.value_head(self.critic_trunk(x))

    def get_policy_and_value(self, x):
        features = self.critic_trunk(x)
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.value_head(features)

    def get_policy_value_pass(self, x, detach_pass: bool):
        """Policy, value and success logit from a single shared critic trunk."""
        features = self.critic_trunk(x)
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        pass_features = features.detach() if detach_pass else features
        return alpha, beta, self.value_head(features), self.pass_head(pass_features).flatten()

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


def maxrl_weight(pass_probability, order):
    """w_T(p) = (1-(1-p)^T)/p, the order-T truncation of the maximum-likelihood weight.

    Evaluated as -expm1(T*log1p(-p))/p: exact for p -> 0, where the naive form is 0/0,
    and monotonically decreasing from w_T(0)=T to w_T(1)=1.
    """
    return -torch.expm1(order * torch.log1p(-pass_probability)) / pass_probability


def sharpen_weight(pass_probability, order):
    """The exact mirror of the MaxRL weight: w_T^s(p) = (1-p^T)/(1-p) = w_T(1-p).

    Same bounded family as w_T -- [1, T], w^s(0)=1, w^s(1)=T -- but INCREASING in p,
    so it spends gradient where good outcomes are already common. Reusing maxrl_weight
    on the complement is not an approximation: substituting q=1-p into
    -expm1(T*log1p(-q))/q gives -expm1(T*log p)/(1-p) identically, and the caller's
    clamp of p to [PASS_FLOOR, PASS_CEIL] keeps q strictly inside (0, 1).
    """
    return maxrl_weight(1.0 - pass_probability, order)


def maxrl_advantages(advantages, returns, pass_logits, bar, args, generator=None):
    """Reweight (and optionally augment) the GAE advantage with the MaxRL estimator.

    Returns the training advantage, the success indicator (the pass head's BCE target)
    and a diagnostics tensor. Called once per iteration under no_grad, like GAE itself.
    """
    probability = torch.sigmoid(pass_logits).clamp(PASS_FLOOR, PASS_CEIL)
    success = (returns > bar).float()

    if args.maxrl_mode == "off":
        # main() already skips this call for `off`; this makes the estimator itself
        # honour the control arm, so no future refactor can silently reweight it.
        weight = torch.ones_like(probability)
    elif args.maxrl_mode == "sharpen":
        weight = sharpen_weight(probability, args.maxrl_order)
    elif args.maxrl_mode in ("weight", "shuffle"):
        weight = maxrl_weight(probability, args.maxrl_order)
    else:
        raise ValueError(f"unknown maxrl_mode {args.maxrl_mode!r}")
    if args.maxrl_mode == "shuffle":
        # Same multiplier distribution, correlation with the state destroyed. Drawn
        # from the run's shuffle generator so the control is seeded like everything else.
        weight = weight[torch.randperm(weight.numel(), device=weight.device, generator=generator)]

    # Mean-1 normalisation keeps the reallocation separate from a gradient-scale
    # change, which per-minibatch norm_adv would otherwise hide inconsistently.
    training_advantages = advantages * (weight / weight.mean())

    # Signed rank correlation between the multiplier and the state's own advantage,
    # the single number that says which way gradient is being reallocated: negative
    # for MaxRL (toward states that are doing badly), positive for sharpen, ~0 for
    # shuffle. Spearman via ranks, so it is invariant to the weight's scale and shape.
    weight_rank = weight.argsort().argsort().float()
    advantage_rank = advantages.argsort().argsort().float()
    reallocation = torch.corrcoef(torch.stack((weight_rank, advantage_rank)))[0, 1]

    diagnostics = torch.stack((
        bar, probability.mean(), probability.std(),
        (probability < 0.1).float().mean(), (probability > 0.9).float().mean(),
        weight.mean(), weight.max(), success.mean(),
        success.mean() - probability.mean(), reallocation,
    ))
    return training_advantages, success, diagnostics


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, success, args):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling."""
    alpha, beta, newvalue, pass_logits = agent.get_policy_value_pass(
        observations, args.maxrl_pass_detach
    )
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
    # Maximum likelihood on the success event: the pass head's own training signal.
    pass_loss = F.binary_cross_entropy_with_logits(pass_logits, success)
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    if args.maxrl_mode != "off":
        loss = loss + args.maxrl_pass_coef * pass_loss
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac, pass_loss.detach()))
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
        raise ValueError("maxrl_order is a truncation level T >= 1 (T=1 reproduces PPO)")
    if not 0.0 < args.maxrl_quantile < 1.0:
        raise ValueError("maxrl_quantile must lie strictly inside (0, 1)")
    if not 0.0 <= args.maxrl_tau_ema < 1.0:
        raise ValueError("maxrl_tau_ema must lie in [0, 1)")
    if args.maxrl_pass_coef < 0.0:
        raise ValueError("maxrl_pass_coef must be non-negative")
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
        writer.add_text("maxrl", f"mode={args.maxrl_mode}; T={args.maxrl_order}; "
                                 f"bar=q{args.maxrl_quantile} EMA {args.maxrl_tau_ema}; "
                                 f"pass_coef={args.maxrl_pass_coef}; detached_trunk={args.maxrl_pass_detach}")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities, values and success logits in one forward."""
            alpha, beta, value, pass_logits = agent.get_policy_value_pass(observations, True)
            return value.flatten(), agent.action_logprob(alpha, beta, native), pass_logits

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values, success):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns,
                            old_values, success, args)

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
        update_metrics = torch.empty((max_updates, 7), device=device)
        # The success bar is device-resident state: reading it would sync the optimizer path.
        success_bar = torch.zeros((), device=device)
        bar_initialised = False
        maxrl_diagnostics = None
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
                b_values, b_logprobs, b_pass_logits = rollout_statistics(b_obs, b_native)
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
                if args.maxrl_mode == "off":
                    b_success = torch.zeros_like(b_returns)
                else:
                    # An absolute bar on the return target, tracked by EMA so the success
                    # event does not jitter with each batch's quantile estimate. It
                    # ratchets upward with the policy: a self-generated verifier.
                    batch_bar = torch.quantile(b_returns, args.maxrl_quantile)
                    if bar_initialised:
                        success_bar.mul_(args.maxrl_tau_ema).add_(
                            batch_bar, alpha=1.0 - args.maxrl_tau_ema
                        )
                    else:
                        success_bar.copy_(batch_bar)
                        bar_initialised = True
                    b_advantages, b_success, maxrl_diagnostics = maxrl_advantages(
                        b_advantages, b_returns, b_pass_logits, success_bar, args,
                        shuffle_generator,
                    )
                    b_advantages = b_advantages.clone()
                    b_success = b_success.clone()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                            b_success[indices],
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
                metric_tensors["losses/pass_loss"] = last[6]
                # phat_std and the two tail fractions are the transplant's go/no-go test:
                # a flat phat means w_T is constant and the method degenerates to PPO.
                metric_tensors.update(zip((
                    "maxrl/success_bar", "maxrl/phat_mean", "maxrl/phat_std",
                    "maxrl/phat_below_0.1", "maxrl/phat_above_0.9",
                    "maxrl/weight_mean", "maxrl/weight_max",
                    "maxrl/success_rate", "maxrl/calibration_error",
                    "maxrl/reallocation_rho",
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
