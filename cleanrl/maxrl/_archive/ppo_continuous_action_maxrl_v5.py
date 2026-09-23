# MaxRL v5: the pass rate is IMPROVABILITY, not state quality.
#
# The claim being chased is the paper's central one -- RL's E[p] is the first-order
# truncation of ML's log E[p], and the fix is the 1/p weight (arXiv 2602.02710, Eq. 11:
# grad J = E[w(p(x)) grad p(x)]).  Everything in the paper is that weight, so porting it
# to dense-reward control is entirely a question of what plays the role of p.
#
# Three versions answered that question the same way and all three lost:
#   v1  p = learned P(segment return > threshold | s)   6057 vs 7991 control
#   v3  p = pass fraction of a cloned-state rollout group 4948 vs 5793 control
#   v4  p = V(s) itself                                  6266 vs 8466 control
# These are one answer, not three: v1's pass head is a monotone function of V and v3's
# group mean is a noisy V.  v4's falsification arms located the damage exactly -- with
# the weight multiset held fixed, `shuffle` (random assignment) cost 300 while `ml`
# (assignment anti-correlated with V) cost 2200.  Dispersion is nearly free; keying the
# weight to state QUALITY is what hurts, and `invert` beating `shuffle` says the sign is
# backwards.  That is a property of MDPs rather than of MaxRL: the paper reweights over
# an exogenous task distribution x ~ rho, but a policy CHOOSES its state distribution,
# and the policy gradient theorem already prices states by discounted visitation.  Low
# value means "state I am trying to leave", not "hard instance deserving attention".
#
# What the latent-generation model actually asks for.  z is the sampled action, and a
# route succeeds when it beats what the policy already achieves from s.  The pass rate
# is then the improvement REACHABLE at s, for which the non-binary M.4 form is
#
#     p(s) = E_{a~pi}[ (A(s,a))_+ ]
#
# the expected positive part of the advantage.  It satisfies M.4's r >= 0 by
# construction (so v4's reward-shift machinery for positivity is simply unnecessary),
# it needs the critic because A does, it thresholds nothing, and it is near-orthogonal
# to V: a high-value state where the policy has converged has LOW improvability, which
# 1/V prices exactly backwards.  That orthogonality is the whole bet, so corr(w, V) is
# logged as the go/no-go -- if it comes back strongly negative this is v4 wearing a hat.
#
# Estimating it.  One sample per state, so a head amortises p(s) across states exactly
# as the critic amortises E[G|s] -- the same "critic as amortiser" move that motivated
# v4, aimed at the right quantity.  The head is trained on relu of the batch-
# standardised advantage (scale-free, so lambda keeps its meaning as the policy's
# reward scale drifts) and it is READ before the update and TRAINED after it, so the
# weight never sees the realisation of the action it will weight.  w must be
# state-only or the estimator is biased for the log objective.
#
# Hypothesis.  If improvability is the right pass rate, `ml` beats the control while
# `invert` loses to it -- the opposite of what v4 measured against V.  If the upside
# head cannot predict (A)_+ from the state at all (`maxrl/upside_ev` ~ 0) then there is
# no substrate and every arm is a relabelled `shuffle`; that reading comes first.
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

    # MaxRL: inverse-value weighting of the policy gradient.
    maxrl_mode: str = "ml"
    """off (plain PPO), ml (1/V weighting), shuffle (dispersion control), invert (sign control)"""
    maxrl_lambda: float = 0.3
    """RL<->ML dial: infinity is uniform weighting, 0 is exact 1/p; holds the denominator positive"""
    upside_coef: float = 0.5
    """weight of the regression that amortises E[(A)_+ | s]"""
    adv_norm_scope: str = "batch"
    """batch (standardise once, preserves reweighting), minibatch (classic PPO), none"""
    maxrl_rescale: bool = True
    """restore the batch advantage scale after weighting so the clip range stays comparable"""

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


def maxrl_weights(pass_rate, args):
    """Inverse-pass-rate weights over the improvement reachable at each state.

    ``pass_rate`` is E[(A)_+ | s], non-negative by construction, so unlike v4's value
    denominator there is no floor to subtract and no reward shift to justify.  lambda
    holds the denominator at or above ``lambda * mean(p) > 0``; the realised ratio is
    ``1 + kappa/lambda`` for the batch's own dispersion kappa = max(p)/mean(p), which
    is why the ratio is logged rather than assumed.
    """
    scale = pass_rate.mean().clamp_min(1e-6)
    weights = (pass_rate + args.maxrl_lambda * scale).reciprocal()
    return weights / weights.mean()


def maxrl_controls(weights, pass_rate, mode, generator):
    """Falsification arms that hold the weight multiset fixed and move only its owner."""
    if mode == "shuffle":
        permutation = torch.randperm(weights.numel(), device=weights.device, generator=generator)
        return weights[permutation]
    if mode == "invert":
        order = pass_rate.argsort()
        reversed_weights = torch.empty_like(weights)
        reversed_weights[order] = weights[order.flip(0)]
        return reversed_weights
    return weights


def upside_targets(advantages):
    """Regression target for the head: relu of the batch-standardised advantage.

    Standardising first makes the target scale-free, so the head's job stays stationary
    while the reward normaliser's units drift and lambda keeps one meaning all run.
    """
    reference = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return reference.clamp_min(0.0)


def maxrl_advantages(advantages, weights, args):
    """Standardise over the whole batch, reweight, then restore the batch scale."""
    if args.maxrl_mode == "off" and args.adv_norm_scope != "batch":
        return advantages
    reference = advantages
    if args.adv_norm_scope == "batch":
        reference = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if args.maxrl_mode == "off":
        return reference
    weighted = reference * weights
    # mean(A*w) = Cov(A, w), whose population value is zero because E[A|s] = 0, but whose
    # sample value is positive for `ml`, ~0 for `shuffle` and negative for `invert` --
    # exactly the ordering the hypothesis predicts. Left in, it would be a nuisance
    # advantage offset (a uniform pull toward or away from the sampled actions) that the
    # falsification arms could not be distinguished from. Re-centring removes it.
    weighted = weighted - weighted.mean()
    if args.maxrl_rescale:
        weighted = weighted * (reference.std() / (weighted.std() + 1e-8))
    return weighted


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
        # initialisation stream as the baseline, so `maxrl_mode=off` (which never
        # backpropagates into this head) stays bit-identical to plain PPO.
        self.upside = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_upside(self, x):
        """E[(A)_+ | s] >= 0. Softplus, because a negative expected positive part is not
        a quantity; this is also why v4's reward shift has no analogue here."""
        return F.softplus(self.upside(x)).flatten()

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
             old_values, upside_target, args):
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
    # Gated at trace time: in `off` the head is absent from the graph, gets no gradient,
    # and the update is bit-identical to plain PPO.
    if args.maxrl_mode == "off":
        upside_loss = torch.zeros((), device=loss.device)
    else:
        upside_loss = ((agent.get_upside(observations) - upside_target) ** 2).mean()
        loss = loss + args.upside_coef * upside_loss
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac, upside_loss.detach()))
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
    if args.maxrl_mode not in {"off", "ml", "shuffle", "invert"}:
        raise ValueError(f"unknown maxrl_mode {args.maxrl_mode!r}")
    if args.adv_norm_scope not in {"batch", "minibatch", "none"}:
        raise ValueError(f"unknown adv_norm_scope {args.adv_norm_scope!r}")
    if args.upside_coef < 0.0:
        raise ValueError("upside_coef must be non-negative")
    if args.maxrl_lambda <= 0.0:
        raise ValueError("maxrl_lambda must be positive; it is the floor of the weight denominator")
    # Per-minibatch standardisation re-standardises random subsets of the reweighted
    # advantages, which partially undoes the reweighting the method exists to apply.
    if args.maxrl_mode != "off" and args.adv_norm_scope == "minibatch":
        raise ValueError("minibatch advantage normalization cancels MaxRL weighting; use adv_norm_scope=batch")
    # Derived, never a flag: a --norm-adv the caller believed they set would be silently
    # discarded here and the hyperparameter dump would show the derived value instead.
    args.norm_adv = args.adv_norm_scope == "minibatch"
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
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native)

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values, upside_target):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns,
                            old_values, upside_target, args)

        upside_model = agent.get_upside

        if args.compile:
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            upside_model = torch.compile(upside_model, fullgraph=True, dynamic=True,
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
        # Separate stream: the shuffle arm must not shift the minibatch order, or the
        # control and the treatment stop sharing a data ordering (a v2 defect).
        weight_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
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
                b_upside_target = upside_targets(b_advantages)
                # Read before the update, trained after it: the weight for a state never
                # sees the realisation of the action it is about to weight.
                pass_rate = upside_model(b_obs) if args.maxrl_mode != "off" \
                    else torch.ones_like(b_advantages)
                weights = maxrl_weights(pass_rate, args)
                weights = maxrl_controls(weights, pass_rate, args.maxrl_mode, weight_generator)
                b_advantages = maxrl_advantages(b_advantages, weights, args)
                # Participation ratio: the fraction of the batch effectively carrying
                # gradient. 1.0 is uniform weighting; lower means concentrated. Unlike a
                # weight/advantage correlation it is not zero by construction.
                participation = weights.sum() ** 2 / (weights.numel() * (weights ** 2).sum())
                # Arm-identity check only, NOT evidence: w is monotone in V and V tracks
                # instantaneous reward, so this is negative for ml, ~0 for shuffle and
                # positive for invert by construction. weight_ratio and participation
                # are the metrics that carry information about the dose.
                centred_w = weights - weights.mean()
                centred_v = b_values - b_values.mean()
                # GO/NO-GO. The bet is that improvability is orthogonal to state quality.
                # Strongly negative here means this has collapsed back into v4's 1/V.
                weight_value_rho = (centred_w * centred_v).mean() / (
                    centred_w.std(unbiased=False) * centred_v.std(unbiased=False) + 1e-12)
                # Is (A)_+ predictable from the state at all? Near zero means the weight
                # is noise and every arm is a relabelled `shuffle`.
                upside_residual = b_upside_target - pass_rate
                upside_ev = 1.0 - upside_residual.var() / b_upside_target.var().clamp_min(1e-8)
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                            b_upside_target[indices],
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
                "losses/upside_loss": update_metrics[:updates, 6].mean(),
                "maxrl/pass_rate_mean": pass_rate.mean(), "maxrl/pass_rate_std": pass_rate.std(),
                "maxrl/weight_ratio": weights.max() / weights.min().clamp_min(1e-12),
                "maxrl/weight_std": weights.std(),
                "maxrl/participation": participation,
                "maxrl/weight_value_rho": weight_value_rho,
                "maxrl/upside_ev": upside_ev,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in ("losses/explained_variance", "maxrl/weight_value_rho",
                                   "maxrl/upside_ev")):
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
