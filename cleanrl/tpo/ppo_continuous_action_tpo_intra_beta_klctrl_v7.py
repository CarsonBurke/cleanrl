# Intra-trajectory TPO with a dual-solved eta under closed-loop KL control. v7.
#
# v7 = v6 plus an outer controller that makes the *realized* policy move, rather
# than the target's KL, the quantity being held constant. Setting
# --realized-kl-target None disables the controller and recovers v6 exactly.
#
# Why v6 was not enough. v6 solves eta so that the TPO target sits kl_budget nats
# from the behaviour policy, which is the right thing to control if the policy can
# reach its target. It cannot. A Beta policy has two parameters per action
# dimension and is asked to hit 2048 per-sample density targets that are mutually
# inconsistent under that family, so it lands on a projection of the target and the
# realized KL falls short. Measured shortfall on this codebase:
#   v6 dyneta  budget 0.125          -> realized approx_kl 0.0657  (0.53)
#   v6 dyneta  budget 0.250          -> realized approx_kl 0.1260  (0.50)
#   v2 relusq  eta=2, budget 0.1225  -> realized approx_kl 0.0391  (0.32)
#   v2         eta=4, budget 0.0309  -> realized approx_kl 0.0033  (0.11)
# The ratio moves 5x across settings and architectures. It is also not a clean
# "fraction of the demanded move": the budget is a joint KL over (s,a) which, with
# one action sample per state, is largely state reweighting, whereas approx_kl is
# the per-state action KL averaged over states -- and the logged ratio does exceed
# 1 at times, which a true realisability fraction could not. Either way, a fixed
# target-KL budget does not pin down the actual step size, and the trust region has
# to mean something about the policy that ends up being deployed.
#
# What v7 adds. After each update, measure the realized KL as the mean approx_kl
# over the final epoch's minibatches -- since the minibatches partition the batch,
# that average is the full-batch KL of the post-update policy against the behaviour
# policy, not a single noisy minibatch. Then drive the target-KL budget by
# proportional control in log space:
#   budget <- budget * exp(-gain * clip(log(realized / realized_target), +/-log c))
# The inner dual solve is unchanged, so the budget still maps to eta correctly for
# any advantage shape; the controller only decides how large a budget to ask for.
# The log-ratio clip bounds how far one anomalous iteration can move the budget,
# and the budget itself is clamped to a sane range.
#
# This composes rather than conflicts: the dual handles distribution shape (fast,
# per-iteration, exact) and the controller handles the model's inability to reach
# its target (slow, integrated over iterations). One knob survives, and it is the
# one with an operational meaning -- how far the policy is allowed to actually move
# per update, in nats.
#
# Note on the setting. The best run measured here (fixed eta=2, 4558 return at
# 1.34M) sits at realized 0.0488, and it beats eta=4 at realized 0.0159 by a wide
# margin, so on HalfCheetah the useful range has been *above* the default target
# used here. The default is deliberately conservative; --realized-kl-target is the
# knob to sweep and it now means exactly what it says.
#
# Why eta should not be a hyperparameter at all. The anchored TPO target
#   q(a|s) ∝ pi_old(a|s) · exp(u(s,a) / eta)
# is not merely inspired by a trust region: it is the exact closed-form solution of
#   maximise E_q[u]   subject to   KL(q ‖ pi_old) <= eps,
# with eta the Lagrange multiplier of that constraint (REPS / MPO E-step). So eta
# is *determined* by the size of the step you want, not chosen independently of it.
# Fixing eta and reading off whatever KL results inverts the causality, and the
# implied step then drifts with the shape of the advantage distribution: after
# whitening, Var(u) = 1 by construction, but
#   KL(eta) = E_w[u]/eta - log E_pi_old[exp(u/eta)] = 1/(2·eta²) + kappa_3/(6·eta³) + ...
# still moves with the skew and kurtosis of u, and with how hard utility_clip
# truncates the tails. Fixed eta therefore silently anneals the trust region as
# the advantage distribution changes over training.
#
# What v6 does. Each iteration, after whitening and clipping the rollout's
# utilities, bisect for the eta whose target sits exactly `kl_budget` nats from the
# behaviour policy. The dual g(eta) = eta·eps + eta·log E[exp(u/eta)] is convex with
# g'(eta) = eps - KL(eta), and KL(eta) is strictly decreasing in eta, so the
# stationary point of the dual is the unique root of KL(eta) = eps and a monotone
# bisection in log-eta finds it safely. Cost is ~40 reductions over 2048 scalars
# per iteration against 320 minibatch updates: free.
#
# The knob becomes a KL budget in nats, which is measurable, comparable across
# environments, and -- the reason this matters for the LLM endgame -- survives
# advantage distributions that are nowhere near Gaussian. Group-normalised binary
# rewards give bimodal utilities for which the 1/(2·eta²) mapping is simply wrong,
# so a fixed eta tuned on one task transfers to another only by luck.
#
# Diagnostics worth watching. tpo/eta_solved shows how much the temperature that
# fixed-eta runs pinned down actually wants to move. tpo/kl_efficiency is realized
# approx_kl divided by the solved target KL: v2 measured 0.52 at eta=4 and 0.75 at
# eta=2, i.e. the Beta family cannot reach the target and the shortfall is itself
# eta-dependent. If that ratio is stable here, the budget is controlling the real
# policy move; if it is not, the honest fix is an outer controller on realized KL.
#
# Hypothesis: replacing fixed eta with a fixed KL budget matches or beats fixed
# eta=2 on HalfCheetah-v4, and removes the need to retune the temperature at all.
#
# v1 recap. Every timestep is its own TPO problem over the binary outcome
# "the executed action vs. the entire rest of the action space" — the continuous
# generalisation of the paper's Appendix-C one-sampled-action construction. Give
# the executed action a cell of width delta per action dimension so it carries
# genuine mass p_old = pi_old(a_t|s_t)·delta^d; the anchored TPO target is
#   q_t = sigmoid(logit(p_old) + u_t/eta),
# fit by binary cross entropy against p_theta = pi_theta(a_t|s_t)·delta^d. For any
# sensible cell p_old, p_theta << 1 and the BCE reduces exactly to
#   grad L_t = delta^d·pi_old(a_t) · (r_t - w_t) · grad log pi_theta(a_t|s_t).
# Dropping the per-sample prefactor weights every trajectory position equally
# (the continuous analogue of "mean over tokens, then over trajectories") and
# leaves a delta-free loss with the same gradient:
#   L_t = w_t·(x - log x - 1),  x = r_t/w_t,  r_t = ratio,  w_t = exp(u_t/eta)
#   grad L_t = (r_t - w_t)·grad log pi_theta(a_t|s_t).
# eta is the trust region: u_t is the rollout-whitened advantage, in units of
# advantage standard deviations, so eta is how many sigma of advantage are needed
# to demand an e-fold change in the executed action's probability. The largest
# demanded change is exp(utility_clip/eta), versus PPO's uniform 1+clip_coef.
#
# What changed in v2.
#  1. Beta action policy. alpha, beta = 1 + softplus(head(s)) per dimension, and
#     the action is an affine map of z ~ Beta(alpha, beta) onto the action box.
#     The support is exactly the action box, so unlike a Gaussian + ClipAction
#     there is no boundary mass that the density fails to account for. This
#     matters more for TPO than for PPO: the target q_t is defined through the
#     behaviour density, so a density that misdescribes what the environment
#     actually executed corrupts the target itself, and TPO has no clip to
#     absorb the error. alpha, beta >= 1 keeps the density unimodal.
#     z is stored in the rollout and replayed directly, so the ratio is always
#     taken between two evaluations of the same density at the same variate.
#     Hidden activations are relu(x)^2 rather than tanh, following the repo's
#     ReluSq convention (same orthogonal init gains).
#  2. Correct overflow guard. v1 clamped logratio inside *both* the exp and the
#     linear -w·log r term, which zeroed the gradient entirely outside the guard
#     — turning the advertised self-correcting trust region into a permanent dead
#     zone in both directions, reachable by tail samples in a 6-dim action space.
#     v2 linearises exp above the guard and never clamps the linear term, so the
#     restoring -w gradient survives for arbitrarily suppressed samples.
#  3. Honest diagnostics (kept, and extended in v6). Residual/movefrac are averaged over whole epochs rather
#     than read off single minibatches, and the first-epoch residual is taken
#     after the first epoch's updates rather than before any step (where ratio is
#     identically 1 and the number carries no information).
#
# Advantages are whitened once over the full rollout, not per minibatch, so the
# target q_t is a single fixed quantity for the whole update as TPO requires.
#
# Hypothesis: on HalfCheetah-v4, exchanging the clipped PPO surrogate for the
# intra-trajectory TPO target and the clipped Gaussian for a Beta policy whose
# support matches the action box improves return, because the TPO target is only
# as good as the behaviour density it anchors to.

import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter


SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """CUDA is required for this experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    compile: bool = False
    """compile the action and value functions with torch.compile"""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
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
    realized_kl_target: float | None = 0.03
    """the actual trust region, in nats: the KL the *post-update policy* is held to
    against the behaviour policy, measured as the final epoch's mean approx_kl.
    The target-KL budget below is driven to achieve this. None disables the
    controller and holds kl_budget fixed, which recovers v6"""
    kl_budget: float = 0.08
    """KL(target ‖ behaviour) over the pooled rollout, in nats; eta is solved from
    this each iteration. Under the controller this is only the starting value.
    0.08 is roughly the budget that yields a realized 0.03 at the 0.39 efficiency
    measured for this architecture, so the controller starts near its fixed point"""
    kl_budget_gain: float = 0.3
    """proportional gain of the log-space budget controller. At 0.3 a persistent
    2x error is corrected in ~8 iterations, far faster than the reward timescale
    and far slower than the per-rollout noise in realized KL"""
    kl_ratio_clip: float = 3.0
    """bound on realized/target per iteration before it enters the controller, so
    one anomalous rollout cannot move the budget by more than gain*log(c)"""
    kl_budget_bounds: tuple[float, float] = (1e-4, 1.0)
    """hard clamp on the controlled budget"""
    max_target_ratio: float = 10.0
    """hard cap on the largest probability change any single sample may demand,
    imposed as a floor eta >= utility_clip / log(cap) on the solved temperature.

    A pooled KL budget bounds the *aggregate* move but not the per-sample gradient
    coefficient, and the two come apart badly on heavy-tailed advantages: whitening
    divides by an outlier-inflated std, which compresses the bulk, so the post-clip
    std collapses and the solver shrinks eta to recover the budget. Measured on
    synthetic rollouts at budget 0.25: gaussian gives eta 1.40 and ratio 8.6, but a
    single +300 outlier among 2047 N(0,1) gives eta 0.62 and ratio 126. The live v6
    kl025 run hit eta 0.496 and a target ratio of 425, at which point a handful of
    samples own the whole grad-norm-clipped update. Worse, at eta near eta_min the
    fp32 gradient norm overflows to inf inside clip_grad_norm_, whose clip
    coefficient is then 0 -- the entire iteration silently becomes a no-op.

    The default caps the ratio at 10 (eta >= 1.303 for utility_clip 3), which is
    looser than fixed eta=2's constant 4.48 and so only bites on the excursions."""
    eta_min: float = 0.05
    """floor on the eta bisection bracket, below the max_target_ratio floor that
    normally binds first. Reached only if utility_clip is None"""
    eta_max: float = 50.0
    """upper bracket for the eta bisection; returned when the batch carries no
    usable advantage signal, which makes the actor exactly neutral"""
    eta_solver_iters: int = 40
    """bisection steps in log-eta; 40 resolves the bracket to ~1e-11 relative"""
    utility_clip: float | None = 3.0
    """clip whitened utilities to +/- this many standard deviations, bounding the
    target ratio to exp(utility_clip / eta); None disables clipping entirely"""
    logratio_guard: float = 20.0
    """log-ratio above which exp() is linearised, purely to keep the loss and its
    gradient finite. Unlike a clip this preserves a monotone, non-zero gradient,
    and the restoring -target gradient below the guard is never affected"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    vloss_clip_coef: float = 0.2
    """the value-function clipping coefficient (TPO has no policy clipping)"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def tpo_utility(advantages: torch.Tensor, utility_clip: float | None) -> torch.Tensor:
    """Whiten advantages into TPO utilities (the paper's `tpo_skill`), then clip.

    A batch with no advantage variance maps to all-zero utility, making the actor
    exactly neutral on it — the continuous analogue of leaving a prompt group with
    no reward variance untouched.
    """
    centered = advantages - advantages.mean()
    # Biased std keeps a length-1 input finite instead of NaN.
    std = centered.std(unbiased=False)
    utility = torch.where(std > 1e-6, centered / std.clamp_min(1e-6), centered)
    if utility_clip is not None:
        utility = utility.clamp(-utility_clip, utility_clip)
    return utility


def tpo_target_kl(utility: torch.Tensor, eta: float) -> float:
    """KL(target ‖ behaviour) for the anchored TPO target at temperature ``eta``.

    With one action sample per state the per-state expectation over actions is a
    single draw, so the batch is pooled into one Monte-Carlo expectation over
    (s, a) ~ pi_old, exactly as in REPS. Writing ``w_i ∝ exp(u_i / eta)`` for the
    self-normalised importance weights that carry pi_old to the target,

        KL(eta) = E_w[u] / eta - log E_pi_old[exp(u / eta)].

    Strictly decreasing in ``eta``: it tends to 0 as eta grows (w becomes uniform
    and E_w[u] tends to the whitened mean, 0) and to log(N / n_max) as eta shrinks
    (w collapses onto the maximum-utility samples).
    """
    z = utility / eta
    log_partition = torch.logsumexp(z, 0) - math.log(utility.numel())
    weights = torch.softmax(z, 0)
    return float((weights * utility).sum() / eta - log_partition)


def tpo_solve_eta(
    utility: torch.Tensor,
    kl_budget: float,
    eta_min: float,
    eta_max: float,
    iters: int,
) -> tuple[float, float]:
    """Solve ``KL(eta) = kl_budget`` for the TPO temperature. Returns (eta, KL).

    This is the stationary point of the convex dual
    ``g(eta) = eta * kl_budget + eta * log E_pi_old[exp(u / eta)]``, since
    ``g'(eta) = kl_budget - KL(eta)``. Because ``KL`` is monotone, bisecting it
    directly is equivalent to minimising ``g`` and cannot be tripped up by the
    dual's flatness near the optimum. Bisection runs in log-eta so the bracket
    spans orders of magnitude in a fixed number of steps.

    Both brackets saturate rather than erroring, and in neither case is the
    returned KL guaranteed to equal the budget, so callers that care must compare.
    ``eta_min`` means the budget exceeds what the clipped utilities can express and
    the returned KL is *below* budget. ``eta_max`` means the budget is smaller than
    even a near-uniform tilt produces, and the returned KL is *above* budget: the
    trust region is being violated, not merely approached. Both are logged.
    """
    if not (kl_budget > 0.0):
        raise ValueError(f"kl_budget must be positive, got {kl_budget}")
    if not eta_min < eta_max:
        raise ValueError(f"empty eta bracket [{eta_min}, {eta_max}]")
    utility = utility.detach().double()
    if not torch.isfinite(utility).all():
        raise ValueError("non-finite utilities reached the eta solver")
    # KL is identically 0 for *constant* utilities, not merely zero ones, so no
    # root exists and bisection would run away to the aggressive bracket. Test the
    # spread, which is the condition that actually matches.
    if float(utility.std(unbiased=False)) <= 1e-12:
        return eta_max, 0.0
    lo, hi = math.log(eta_min), math.log(eta_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if tpo_target_kl(utility, math.exp(mid)) > kl_budget:
            lo = mid  # too aggressive, eta must grow
        else:
            hi = mid
    eta = math.exp(0.5 * (lo + hi))
    return eta, tpo_target_kl(utility, eta)


def eta_floor(utility_clip: float | None, max_target_ratio: float, eta_min: float) -> float:
    """Smallest eta whose most extreme sample still demands at most ``max_target_ratio``.

    ``exp(utility_clip / eta) <= cap`` rearranges to ``eta >= utility_clip / log cap``.
    With no utility clip there is no largest sample and no such floor exists, so
    the raw bracket stands.
    """
    if utility_clip is None or not math.isfinite(max_target_ratio):
        return eta_min
    if max_target_ratio <= 1.0:
        raise ValueError(f"max_target_ratio must exceed 1, got {max_target_ratio}")
    return max(eta_min, utility_clip / math.log(max_target_ratio))


def kl_budget_update(
    budget: float,
    realized_kl: float,
    realized_kl_target: float | None,
    gain: float,
    ratio_clip: float,
    bounds: tuple[float, float],
) -> float:
    """Proportional control of the target-KL budget on ``log(realized / target)``.

    Log space rather than a difference, so the correction is relative and the
    controller behaves the same whether the budget is 0.3 or 0.003. The response is
    monotone: overshooting the realized target shrinks the budget, which raises the
    solved eta, which demands a smaller move.

    ``ratio_clip`` bounds a single iteration's influence to ``exp(gain*log c)``, so
    a rollout whose realized KL is wild -- a near-degenerate advantage batch, an
    epoch cut short by target_kl -- cannot slam the budget into a bound and strand
    it there for the many iterations it would take to walk back out.
    """
    if realized_kl_target is None:
        return budget
    log_ratio = math.log(max(realized_kl, 1e-8) / realized_kl_target)
    log_ratio = min(max(log_ratio, -math.log(ratio_clip)), math.log(ratio_clip))
    lo, hi = bounds
    return min(max(budget * math.exp(-gain * log_ratio), lo), hi)


def tpo_intra_loss(
    logratio: torch.Tensor,
    utility: torch.Tensor,
    eta: float,
    logratio_guard: float,
) -> torch.Tensor:
    """Per-sample intra-trajectory TPO loss with gradient (ratio - target).

    ``L = w * (x - log x - 1)`` for ``x = r/w``, ``r = exp(logratio)`` and target
    ratio ``w = exp(utility / eta)``. Non-negative, convex in ``logratio``, and
    zero exactly at ``r = w``, so ``utility = 0`` gives zero gradient at ``r = 1``.

    Only ``exp`` is guarded, and by linearisation rather than clamping: above
    ``logratio_guard`` the gradient saturates at ``exp(logratio_guard) - w``
    instead of dropping to zero. The linear ``-w * logratio`` term is never
    guarded, so a heavily suppressed sample keeps its full restoring ``-w``
    gradient no matter how far it has been pushed.
    """
    log_target_ratio = utility.detach() / eta
    target_ratio = log_target_ratio.exp()
    guard_ratio = math.exp(logratio_guard)
    ratio = torch.where(
        logratio > logratio_guard,
        guard_ratio * (logratio - logratio_guard + 1.0),
        logratio.clamp_max(logratio_guard).exp(),
    )
    return ratio - target_ratio * logratio - target_ratio + target_ratio * log_target_ratio


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReluSq(nn.Module):
    """f(x) = relu(x)^2."""

    def forward(self, x):
        return torch.relu(x).square()


class Agent(nn.Module):
    """Beta policy on the native action box; critic unchanged from the baseline."""

    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
        )
        self.actor_alpha = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def get_value(self, x):
        return self.critic(x)

    def _dist(self, x):
        h = self.actor(x)
        # alpha, beta >= 1 keeps the density unimodal and finite at the edges.
        alpha = 1.0 + F.softplus(self.actor_alpha(h))
        beta = 1.0 + F.softplus(self.actor_beta(h))
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def get_action_and_value(self, x, z=None):
        """Return the executed action alongside the density of the *stored* variate.

        z (the Beta variate) is what the rollout stores, so the ratio is always a
        comparison of two densities evaluated at the same point; round-tripping
        through the action and back would lose information to the edge clamp.
        """
        dist = self._dist(x)
        if z is None:
            z = dist.sample()
        z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        logprob = dist.log_prob(z).sum(1)
        concentration = (dist.concentration1 + dist.concentration0).mean()
        return self._z_to_action(z), z, logprob, dist.entropy().sum(1), self.critic(x), concentration


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size % args.num_minibatches:
        raise ValueError(
            f"num_minibatches ({args.num_minibatches}) must divide batch_size ({args.batch_size}); "
            "a ragged final minibatch breaks static shapes under --compile"
        )
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        raise ValueError("this experiment requires CUDA; --no-cuda is unsupported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")
    device = torch.device("cuda")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert np.all(np.isfinite(envs.single_action_space.low)) and np.all(
        np.isfinite(envs.single_action_space.high)
    ), "a Beta policy requires a bounded action space"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    action_and_value = agent.get_action_and_value
    value_function = agent.get_value
    if args.compile:
        action_and_value = torch.compile(action_and_value, mode=args.compile_mode, dynamic=False)
        value_function = torch.compile(value_function, mode=args.compile_mode, dynamic=False)
        print(f"compiled action and value functions (mode={args.compile_mode!r}, dynamic=False)")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # Controller state: the target-KL budget the dual solve is asked for. Carried
    # across iterations because the quantity it corrects for -- how much of the
    # demanded move the Beta family can realise -- drifts slowly with training.
    kl_budget = args.kl_budget
    # A pooled KL budget does not bound the per-sample target ratio; this floor does.
    solver_eta_min = eta_floor(args.utility_clip, args.max_target_ratio, args.eta_min)
    print(f"eta bracket [{solver_eta_min:.4f}, {args.eta_max}] caps the target ratio at {args.max_target_ratio}")

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                action, z, logprob, _, value, _ = action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            zs[step] = z
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            next_value = value_function(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ALGO LOGIC: the TPO target is fixed for the whole update, so utilities
        # are whitened once over the rollout rather than per minibatch. The
        # temperature is then solved from those exact utilities -- post-clip,
        # because the clipped values are what the target is actually built from --
        # so the target sits kl_budget nats from the behaviour policy by
        # construction rather than by whatever a fixed eta happens to imply.
        b_utilities = tpo_utility(b_advantages, args.utility_clip)
        eta, solved_kl = tpo_solve_eta(
            b_utilities, kl_budget, solver_eta_min, args.eta_max, args.eta_solver_iters
        )

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        residuals = []
        approx_kls = []
        movefracs = []
        coef_maxes = []
        concentrations = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                _, _, newlogprob, entropy, newvalue, concentration = action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.clamp_max(args.logratio_guard).exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls += [approx_kl.item()]
                    movefracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]
                    concentrations += [concentration.item()]

                mb_utilities = b_utilities[mb_inds]

                # Policy loss: fit the executed action's probability to the
                # anchored TPO target pi_old * exp(u / eta). No clipping.
                pg_loss = tpo_intra_loss(logratio, mb_utilities, eta, args.logratio_guard).mean()

                with torch.no_grad():
                    # |ratio - target| is the TPO fit error, and it should shrink
                    # across epochs because the target does not move. Its max is
                    # the per-sample gradient coefficient, so a large value means
                    # one sample is dominating the minibatch direction.
                    coefficient = ratio - (mb_utilities / eta).exp()
                    residuals += [coefficient.abs().mean().item()]
                    coef_maxes += [coefficient.abs().max().item()]

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.vloss_clip_coef,
                        args.vloss_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # The minibatches of an epoch partition the batch, so the mean approx_kl
        # over the last epoch is the full-batch KL of the post-update policy
        # against the behaviour policy -- the move that actually happened.
        realized_kl = float(np.mean(approx_kls[-args.num_minibatches :]))
        budget_used = kl_budget
        kl_budget = kl_budget_update(
            kl_budget,
            realized_kl,
            args.realized_kl_target,
            args.kl_budget_gain,
            args.kl_ratio_clip,
            args.kl_budget_bounds,
        )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        n_mb = args.num_minibatches
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/movefrac", np.mean(movefracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # Averaged over whole epochs; the first-epoch value is measured after the
        # first epoch's updates, since before any step the ratio is exactly 1.
        writer.add_scalar("tpo/residual_first_epoch", np.mean(residuals[:n_mb]), global_step)
        writer.add_scalar("tpo/residual_last_epoch", np.mean(residuals[-n_mb:]), global_step)
        writer.add_scalar("tpo/coefficient_max", np.max(coef_maxes), global_step)
        writer.add_scalar("tpo/beta_concentration", np.mean(concentrations), global_step)
        writer.add_scalar(
            "tpo/target_ratio_max",
            float(np.exp((args.utility_clip or np.inf) / eta)),
            global_step,
        )
        # The temperature a fixed-eta run would have had to guess, and the KL it
        # buys. solved_kl tracks kl_budget except when a bracket saturates.
        writer.add_scalar("tpo/eta_solved", eta, global_step)
        writer.add_scalar("tpo/target_kl", solved_kl, global_step)
        # Controller state. realized_kl is the quantity actually being held to
        # target; kl_budget is what the controller had to ask for to get it.
        writer.add_scalar("tpo/realized_kl", realized_kl, global_step)
        writer.add_scalar("tpo/kl_budget", budget_used, global_step)
        # Deliberately not called an efficiency: the numerator is a per-state
        # action KL and the denominator a joint (s,a) KL, so the ratio is a trend
        # line for how the two drift apart, not a realisability fraction.
        writer.add_scalar(
            "tpo/realized_over_budgeted",
            realized_kl / solved_kl if solved_kl > 0.0 else 0.0,
            global_step,
        )
        # Flags the two saturating outcomes, where the returned KL is not the budget.
        writer.add_scalar("tpo/eta_at_floor", float(eta <= solver_eta_min * (1.0 + 1e-6)), global_step)
        writer.add_scalar("tpo/eta_at_ceiling", float(eta >= args.eta_max * (1.0 - 1e-6)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
