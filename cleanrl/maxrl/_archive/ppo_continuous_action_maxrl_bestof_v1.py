# MaxRL-bestof v1: the EXACT best-of-T policy gradient, with PPO as its T=1 member.
#
# Where tail_v1 landed.  Table 2's order-T weight w_T = (1-(1-p)^T)/p is the gradient of
# 1-(1-p)^T = pass@T, so MaxRL's dial is "maximise the best of T attempts" and T=1 is RL.
# tail_v1 implemented that by CONDITIONING on the upper tail -- a hard hinge at the
# alpha-quantile, alpha = 1-1/T.  On HalfCheetah at 8M it lost at every alpha (control
# 8064; alpha .25/.5/.75/.9 = 7312/7455/5624/4958) and collapsed between alpha 0.5 and
# 0.75, i.e. once the FLAT region covered most of the batch.  But its falsification arm
# paid off: the lower tail at the same alpha scored 5873 against the upper tail's 7455, a
# 1580 gap in the predicted direction, and not explained by either diagnostic (the lower
# arm is the CLOSER of the two to PPO in corr(w, A), at 0.89 against 0.60).  Direction
# carries real signal here; the hinge is what threw the score away.
#
# What the hinge got wrong.  E[Z | Z >= q] is only an APPROXIMATION to E[max of T], and a
# hard one: it assigns every sample below the quantile the identical push, discarding how
# far below it fell.  In the paper's setting that costs nothing -- failures are
# indistinguishable, there is no gradation among them to discard.  Dense reward is exactly
# the setting where that gradation exists, so truncating it throws away the information
# this domain uniquely provides.  The paper's own family is SMOOTH in p; the hinge is not.
#
# The exact objective.  Take J_T = E[max(Z_1..Z_T)] directly.  With F the return CDF,
#     grad J_T = E[h(Z) grad log pi],    h'(Z) = T F(Z)^(T-1)
# so the weight is a smooth, monotone function of the sample's PERCENTILE, with slope
# T F^(T-1): near zero low down, steep at the top, and never flat.  T=1 gives slope 1
# everywhere, h(Z) = Z + const, w = A: PPO exactly.  Writing Z|s = V(s) + sigma(s) X,
#     w = sigma(s) * psi_T(A / sigma(s)) - b(s),   psi_T(x) = T * int_0^x Phi(v)^(T-1) dv
# and psi_1 is the identity to machine precision.  psi_T is one fixed 1-D quadrature; the
# critic gains sigma(s) (Gaussian NLL) and b(s) (the conditional mean of the weighted
# score, so w is centred per state whatever the Gaussian fit is worth).
#
# Verified, not assumed.  E[w * grad log p] reproduces d/dmu E[max of T] = 1 and, the
# T-discriminating check, d/dsigma E[max of T] = E[max of T standard normals] (T=2,4,8:
# 0.5651/0.5642, 1.0304/1.0297, 1.4250/1.4232).  `tail/below_frac` calibration in tail_v1
# tracked alpha at every level, so the head machinery itself was never the problem.
#
# Arms.  `--bestof-scale unit` fixes sigma == 1, isolating the rank NONLINEARITY from the
# per-state SCALE -- the per-state reweighting that five earlier versions died on, kept as
# a separate factor so it cannot be confounded with the nonlinearity.  `--bestof-side
# lower` maximises E[min of T] (pessimism) and coincides with PPO at T=1, the falsification
# arm that already discriminated once.
#
# Hypothesis.  If the hinge's flat region was the damage, the smooth weight at T=4 lands
# near control where tail_v1's matched alpha=0.75 lost 2440.  If the smooth weight loses
# just as much, the objective and not its approximation is what this domain rejects.
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
SPREAD_FLOOR = 0.05
# Beyond +-8 standard deviations Phi is 1 or 0 to within 1e-15, so psi_T is exactly linear
# there and the quadrature only has to cover this range.
RANK_LIMIT = 8.0
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
    bestof_scale: str = "state"
    """state uses the learned sigma(s); unit fixes sigma == 1, isolating the nonlinearity"""
    bestof_nodes: int = 128
    """midpoint nodes for the psi_T quadrature over [-8, 8]; the rule is monotone only to
    its own O(h^2) error, ~2e-7 here, which is below the float32 floor the trainer runs at"""
    bestof_coef: float = 0.5
    """weight of the scale and baseline regressions that amortise the per-state statistics"""

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


def rank_weight(x, order, nodes):
    """psi_T(x) = T * int_0^x Phi(v)^(T-1) dv, the exact best-of-T score-function weight.

    Its derivative is T F(x)^(T-1): the slope a sample earns for its PERCENTILE, near zero
    low down and steep at the top, but never flat -- which is the whole difference from the
    hinge tail_v1 used, and why gradation among poor samples survives here.

    Substituting v = x t turns the integral into x * int_0^1 f(x t) dt, one vectorised
    midpoint rule with no branch on the sign of x.  At order 1 the integrand is identically
    one and this returns x exactly, in floating point, which is what makes PPO a member of
    the family rather than a nearby approximation to it.

    That substitution spaces the nodes |x|/nodes apart, so for large |x| it steps straight
    over the region near zero where the integrand actually varies: at x = -1000 and T = 4
    the first node already sits at -7.8 and the rule returns -2e-30 for a true value of
    -0.167.  So integrate only over the clamped range, where the spacing stays below
    2*LIMIT/nodes, and extend beyond it along the exact asymptotic slope T Phi(+-LIMIT)^(T-1)
    -- which is T above (Phi -> 1) and vanishing below (Phi -> 0), and is what keeps order 1
    the identity, since there Phi^0 = 1 makes the extension exactly linear everywhere.
    """
    clamped = x.clamp(-RANK_LIMIT, RANK_LIMIT)
    nodes_t = (torch.arange(nodes, device=x.device, dtype=x.dtype) + 0.5) / nodes
    cdf = 0.5 * (1.0 + torch.erf(clamped.unsqueeze(-1) * nodes_t * 0.7071067811865476))
    edge = 0.5 * (1.0 + torch.erf(clamped * 0.7071067811865476))
    integral = clamped * cdf.pow(order - 1.0).mean(-1)
    return order * (integral + (x - clamped) * edge.pow(order - 1.0))


def bestof_advantages(oriented, scale, baseline, order, nodes):
    """w = sigma(s) psi_T(A/sigma(s)) - b(s), the best-of-T analogue of A = R - V(s).

    ``b`` is the conditional mean of the first term, regressed directly rather than taken
    from the Gaussian model's closed form: that keeps w centred per state even where the
    advantage is not Gaussian, so a poor scale fit costs accuracy but never injects the
    state-dependent offset that would order the arms by a nuisance.
    """
    return scale * rank_weight(oriented / scale, order, nodes) - baseline


def scale_nll(oriented, scale):
    """Gaussian NLL for a zero-mean scale: minimised at sigma^2 = E[A^2 | s].

    E[A|s] = 0 makes the squared advantage an unbiased target for the conditional variance;
    fitting it through the log-likelihood of the very Gaussian the transform assumes is
    better conditioned than regressing sigma^2 on A^2, which is a fourth-power loss.
    """
    return 0.5 * (oriented / scale) ** 2 + scale.log()


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
            layer_init(nn.Linear(64, 2), std=1.0),
        )

    def get_spread(self, x):
        """(scale, baseline) for the batch-standardised advantage at s.

        The scale is a positive dispersion and is softplus-constrained above a floor; the
        baseline is the conditional mean of the weighted score, a location on the real
        line, and is not.  Both are emitted by one network so the two arms of the scale
        experiment share identical head dynamics and differ only in where sigma is consumed.
        """
        head = self.spread(x)
        return F.softplus(head[..., 0]) + SPREAD_FLOOR, head[..., 1]

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

    ``advantages`` already carries the rank transform; ``oriented`` is the oriented,
    batch-standardised advantage the scale head regresses against, and the two coincide at
    T=1.  ``weighted_target`` is sigma psi_T(A/sigma) evaluated once per batch from the
    pre-update head: it is constant in the parameters, so computing it here would run the
    quadrature once per minibatch (320 times an iteration) for an identical number, and
    holding it fixed across the epochs also makes it a stationary regression target rather
    than one that chases the scale head as it moves.
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
        scale, baseline = agent.get_spread(observations)
        # The target carries no gradient into the scale: the baseline regression estimates
        # the conditional mean GIVEN the scale, and must not drag the scale toward whatever
        # value makes its own target easy to fit.
        head_loss = (scale_nll(oriented, scale) + (baseline - weighted_target) ** 2).mean()
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
    if args.bestof_scale not in {"state", "unit"}:
        raise ValueError(f"unknown bestof_scale {args.bestof_scale!r}")
    if args.bestof_nodes < 8:
        raise ValueError("bestof_nodes below 8 makes the psi_T quadrature the dominant error")
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
        writer.add_text("objective", "best-of-T: w = sigma(s) psi_T(A/sigma(s)) - b(s); psi_T' = T Phi^(T-1); T=1 is PPO")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
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
                    scale = torch.ones_like(b_advantages)
                else:
                    # Standardised over the whole batch so the head's targets carry fixed
                    # units while the reward normaliser's scale drifts.  Minibatch
                    # standardisation in the loss is invariant to this global affine map,
                    # so it changes nothing the policy sees beyond fixing the head's units.
                    standardised = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                    b_oriented = sign * standardised
                    # Read before the update and trained after it, so the statistics a
                    # state is judged against never see the action actually taken there.
                    scale, baseline = spread_model(b_obs)
                    if args.bestof_scale == "unit":
                        scale = torch.ones_like(scale)
                    b_weighted_target = scale * rank_weight(b_oriented / scale,
                                                             args.bestof_order, args.bestof_nodes)
                    b_weights = sign * (b_weighted_target - baseline)
                # Is the Gaussian model the transform assumes actually met?  psi_T reads
                # A/sigma as a z-score, so a kurtosis far from 3 means the percentiles it
                # implies are wrong and the realised order is not the requested T.  This is
                # the GO/NO-GO, and it is a fact about the environment, not about the head.
                normalised = b_oriented / scale
                kurtosis = (normalised ** 4).mean() / (normalised ** 2).mean().clamp_min(1e-12) ** 2
                # Does sigma vary across states at all, or has it collapsed to a constant?
                # A collapsed scale makes `state` a relabelled `unit`, and the two arms
                # would then be compared as though they differed when they do not.
                scale_dispersion = scale.std() / scale.mean().clamp_min(1e-12)
                # Held-out centring: w must have conditional mean zero for the baseline
                # argument to hold, and this is read before the head is trained on it.
                weight_mean = b_weights.mean() / b_weights.std().clamp_min(1e-12)
                # How far the transform moved from PPO, and what it cost in variance.
                centred_w = b_weights - b_weights.mean()
                centred_a = b_advantages - b_advantages.mean()
                weight_adv_rho = (centred_w * centred_a).mean() / (
                    centred_w.std(unbiased=False) * centred_a.std(unbiased=False) + 1e-12)
                weight_std_ratio = b_weights.std() / b_advantages.std().clamp_min(1e-12)
                # Realised dose: the slope T Phi(x)^(T-1) earned at the top decile.  At T=1
                # this is 1 everywhere; it is the effective aggressiveness of the arm.
                top = torch.quantile(normalised, 0.9)
                top_slope = args.bestof_order * (0.5 * (1.0 + torch.erf(
                    top * 0.7071067811865476))) ** (args.bestof_order - 1.0)
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
                "losses/head_loss": update_metrics[:updates, 6].mean(),
                "bestof/kurtosis": kurtosis,
                "bestof/scale_dispersion": scale_dispersion,
                "bestof/scale_mean": scale.mean(),
                "bestof/weight_mean": weight_mean,
                "bestof/weight_adv_rho": weight_adv_rho,
                "bestof/weight_std_ratio": weight_std_ratio,
                "bestof/top_slope": top_slope,
            })
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in ("losses/explained_variance", "bestof/weight_adv_rho",
                                   "bestof/weight_mean")):
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
