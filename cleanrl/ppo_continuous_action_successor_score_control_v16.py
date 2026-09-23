# Fixed-task successor-action score control PPO v16.
# All three arms retain v15's scalar Tanh64x64 actor/critic, host sampling, normalization,
# factual-next GAE and clipped PPO. A third Tanh64x64 predicts [b0,b_alpha,b_beta].
# Its frozen behavior-policy control is c(s,u)=b0+b.S_old(s,u), integrated exactly
# under the NEW ideal Beta policy by differences of analytic log moments.
# The coefficient network minimizes actor-PARAMETER gradient second moment, not
# return prediction, latent prediction, SF prediction or an auxiliary critic loss.
# Let J=d(alpha,beta)/d logits, K=Jac_actor Jac_actor^T and
# D=J[S, SS^T-F]. Then g=J S A-D b, E[D|s]=0, and the conditional optimum obeys
# E[D^T K D|s] b=E[D^T K J S A|s]. Training estimates this objective directly;
# it does not form the sample score outer product or solve noisy normal equations.
# This is a reward-contracted successor-action score projection, not a claim that
# generic action-dependent control variates are new. Its exact optimum is scoped
# to ideal-Beta, unclipped, unpreconditioned behavior-policy gradients in the
# chosen advantage units. It is NOT an Adam/clipped-PPO/reused-minibatch global
# optimum or unbiased-GAE claim. Fit targets use PPO's exact normalized minibatch
# advantages, while preupdate diagnostics use full-rollout normalization.
# Adaptive updates and finite-precision epsilon-clipped sampling limit the ideal
# identity. Diagnostics predict BEFORE fitting that rollout, not an in-sample refit.
# Precedents: Q-Prop (https://arxiv.org/abs/1611.02247) and direct variance training
# in LAX/RELAX (https://arxiv.org/abs/1711.00123); this is an analytic Beta adaptation.
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

from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import device_minibatches, explained_variance, gather_metrics, get_gae_fn
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    save_model: bool = False
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    control_mode: Literal["ppo", "baseline", "corrected"] = "corrected"
    """Identical full control fitting; actor uses none, b0 only, or full control."""
    env_backend: Literal["native"] = "native"
    env_threads: int = 2
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    non_blocking_transfers: bool = False
    staggered_starts: bool = True
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TaskFFN(nn.Module):
    """Proper two-stage 64x64 FFN; no shared or detached hidden stage."""

    def __init__(self, input_dim, output_dim, output_std):
        super().__init__()
        self.first = nn.Sequential(layer_init(nn.Linear(input_dim, 64)), nn.Tanh())
        self.second = nn.Sequential(layer_init(nn.Linear(64, 64)), nn.Tanh())
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))

    def forward(self, x):
        return self.head(self.second(self.first(x)))


class HostPolicy:
    """One fused native FP32 actor graph, refreshed once after learner updates."""

    def __init__(self, agent, num_rows):
        self.fused = make_host_mirror(nn.Sequential(
            *agent.actor.first, *agent.actor.second, *agent.actor.head,
        ), num_rows)

    def refresh(self):
        self.fused.refresh()

    def __call__(self, observations):
        return self.fused(observations)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        if args.control_mode not in {"ppo", "baseline", "corrected"}:
            raise ValueError("control_mode must be ppo, baseline or corrected")
        self.control_mode = args.control_mode
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
        # The complete canonical v15 scalar initialization precedes any control RNG.
        self.actor = TaskFFN(observation_dim, 2 * self.action_dim, 0.01)
        self.critic = TaskFFN(observation_dim, 1, 1.0)
        with torch.random.fork_rng(devices=[]):
            self.control = TaskFFN(observation_dim, 1 + 2 * self.action_dim, 0.0)
        self.actor_parameter_count = sum(parameter.numel() for parameter in self.actor.parameters())

    def parameter_groups(self):
        """Three exclusive Adam owners; every trainable parameter is owned once."""
        return tuple(self.actor.parameters()), tuple(self.critic.parameters()), tuple(self.control.parameters())

    def parameter_counts(self):
        actor, critic, control = (sum(parameter.numel() for parameter in group)
                                  for group in self.parameter_groups())
        return {"actor_ffn": actor, "critic_ffn": critic, "control_ffn": control,
                "inference": actor + critic, "total": actor + critic + control}

    def get_value(self, observations):
        return self.critic(observations.detach())

    def get_policy_and_value(self, observations):
        logits = self.actor(observations.detach())
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.get_value(observations)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action.detach()) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training stores unit-interval samples."""
        alpha, beta, value = self.get_policy_and_value(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.detach().reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS,
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


def beta_log_moments(alpha, beta):
    """E[log u, log(1-u)], with all alpha coordinates before all beta ones."""
    total = torch.digamma(alpha + beta)
    return torch.cat((torch.digamma(alpha) - total, torch.digamma(beta) - total), dim=-1)


def beta_score(native, alpha, beta):
    """Concentration-parameter score of independent native Beta actions."""
    return torch.cat((native.log(), torch.log1p(-native)), dim=-1) - beta_log_moments(alpha, beta)


def beta_fisher_components(alpha, beta):
    """Diagonal and within-action cross terms, cached once per behavior rollout."""
    common = torch.polygamma(1, alpha + beta)
    diagonal = torch.cat((torch.polygamma(1, alpha) - common,
                          torch.polygamma(1, beta) - common), dim=-1)
    return diagonal, -common


def cached_fisher_product(diagonal, cross, vector):
    va, vb = vector.chunk(2, dim=-1)
    return diagonal * vector + torch.cat((cross * vb, cross * va), dim=-1)


def beta_fisher_product(alpha, beta, vector):
    """Independent two-coordinate Fisher blocks, without constructing a matrix."""
    diagonal, cross = beta_fisher_components(alpha, beta)
    return cached_fisher_product(diagonal, cross, vector)


def beta_logit_jacobian(alpha, beta):
    """Diagonal softplus derivative recovered from concentrations greater than 1."""
    return torch.cat((-torch.expm1(-(alpha - 1)), -torch.expm1(-(beta - 1))), dim=-1)


@torch.no_grad()
def actor_jacobian_gram(actor, obs):
    """Jac_actor Jac_actor^T for every weight and bias, at detached observations."""
    x = obs.detach()
    h1 = actor.first(x)
    h2 = actor.second(h1)
    r = actor.head[0].weight.unsqueeze(0) * (1 - h2.square()).unsqueeze(-2)
    u = (r @ actor.second[0].weight) * (1 - h1.square()).unsqueeze(-2)
    identity = torch.eye(actor.head[0].out_features, device=x.device, dtype=x.dtype)
    head_term = (h2.square().sum(-1) + 1)[:, None, None] * identity
    second_term = (h1.square().sum(-1) + 1)[:, None, None] * (r @ r.transpose(-1, -2))
    first_term = (x.square().sum(-1) + 1)[:, None, None] * (u @ u.transpose(-1, -2))
    return (head_term + second_term + first_term).detach()


def control_sample(coeff, score):
    return coeff[..., 0] + (coeff[..., 1:] * score).sum(-1)


def control_expectation(coeff, alpha, beta, old_moments):
    """Integrate frozen behavior-score features; ONLY new policy is attached."""
    frozen = coeff.detach()
    return frozen[..., 0] + (frozen[..., 1:] * (
        beta_log_moments(alpha, beta) - old_moments.detach()
    )).sum(-1)


def cached_corrected_output_gradient(advantages, score, coeff, fisher_diagonal, fisher_cross, logit_jacobian):
    """Only coefficients attach; all cached behavior geometry stays immutable."""
    advantages, score = advantages.detach(), score.detach()
    residual = advantages - control_sample(coeff, score)
    response = cached_fisher_product(fisher_diagonal.detach(), fisher_cross.detach(), coeff[..., 1:])
    return logit_jacobian.detach() * (score * residual.unsqueeze(-1) + response)


def corrected_output_gradient(advantages, score, alpha, beta, coeff):
    """Positive objective logit gradient in supplied advantage units.

    At behavior theta: J[S(A-c)+F b]. PPO minimizes the negative objective;
    flipping that sign does not change the gradient variance objective.
    ONLY coefficients remain attached, including when supplied teachers attach.
    """
    alpha, beta = alpha.detach(), beta.detach()
    diagonal, cross = beta_fisher_components(alpha, beta)
    return cached_corrected_output_gradient(
        advantages, score, coeff, diagonal, cross, beta_logit_jacobian(alpha, beta),
    )


def control_variance_loss(coeff, advantages, score, alpha, beta, gram, actor_parameter_count,
                          *, fisher_diagonal=None, fisher_cross=None, logit_jacobian=None):
    """Unpreconditioned actor-gradient second moment per actor parameter.

    The conditional mean is coefficient-independent under ideal Beta sampling,
    so minimizing this second moment minimizes the corresponding variance trace.
    All behavior/GAE/actor geometry teachers are detached, not jointly optimized.
    Cached kwargs omit all special-function evaluation in the training path.
    """
    if fisher_diagonal is None:
        gradient = corrected_output_gradient(advantages, score, alpha, beta, coeff)
    else:
        gradient = cached_corrected_output_gradient(
            advantages, score, coeff, fisher_diagonal, fisher_cross, logit_jacobian,
        )
    squared_norm = (gradient.unsqueeze(-2) @ gram.detach() @ gradient.unsqueeze(-1)).flatten()
    return 0.5 * squared_norm.mean() / actor_parameter_count


@torch.no_grad()
def actor_gradient_moments(actor, obs, output_gradients):
    """Return (E||parameter gradient||^2, ||E parameter gradient||^2).

    Analytic MLP derivatives avoid per-example parameter tensors and autograd.
    The first term is g^T K g factored layerwise, avoiding another cached Gram.
    """
    x, g = obs.detach(), output_gradients.detach()
    h1 = actor.first(x)
    h2 = actor.second(h1)
    d2 = (g @ actor.head[0].weight) * (1 - h2.square())
    d1 = (d2 @ actor.second[0].weight) * (1 - h1.square())
    per_example = (g.square().sum(-1) * (h2.square().sum(-1) + 1)
                   + d2.square().sum(-1) * (h1.square().sum(-1) + 1)
                   + d1.square().sum(-1) * (x.square().sum(-1) + 1))
    rows = x.shape[0]
    mean_squared = sum(((delta.T @ inputs) / rows).square().sum() + delta.mean(0).square().sum()
                       for delta, inputs in ((g, h2), (d2, h1), (d1, x)))
    return per_example.mean().detach(), mean_squared.detach()


def critic_loss(predictions, returns, old_values, args):
    """The unchanged scalar clipped task-value objective."""
    values = predictions.squeeze(-1)
    returns, old_values = returns.detach(), old_values.detach()
    squared_error = (values - returns).square()
    if args.clip_vloss:
        clipped = old_values + (values - old_values).clamp(-args.clip_coef, args.clip_coef)
        squared_error = torch.maximum(squared_error, (clipped - returns).square())
    return 0.5 * squared_error.mean()


def policy_loss(agent, observations, native_actions, advantages, returns, old_data, args):
    """One backward for disjoint actor, scalar critic and control objectives.

    old_data is the prefit rollout_statistics dictionary, minibatch-indexed by
    the caller. Actor correction uses ONLY these frozen coefficients and score
    features; current control predictions participate ONLY in variance fitting.
    """
    alpha, beta, predictions = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions.detach()) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1).mean()
    logratio = newlogprob - old_data["logprobs"].detach()
    ratio = logratio.exp()
    ppo_advantages = advantages.detach()
    if args.norm_adv:
        ppo_advantages = (ppo_advantages - ppo_advantages.mean()) / (ppo_advantages.std() + 1e-8)
    pg_loss = torch.maximum(-ppo_advantages * ratio,
                            -ppo_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    value_loss = critic_loss(predictions, returns, old_data["values"], args)
    coefficients = agent.control(observations.detach())
    variance_loss = control_variance_loss(
        coefficients, ppo_advantages, old_data["score"], old_data["alpha"], old_data["beta"],
        old_data["gram"], agent.actor_parameter_count, fisher_diagonal=old_data["fisher_diagonal"],
        fisher_cross=old_data["fisher_cross"], logit_jacobian=old_data["logit_jacobian"],
    )
    old_coefficients = old_data["coefficients"].detach()
    if args.control_mode == "baseline":
        old_control = expected_control = old_coefficients[..., 0]
    else:
        old_control = old_data["control_samples"].detach()
        expected_control = control_expectation(old_coefficients, alpha, beta, old_data["log_moments"])
    raw_correction = (ratio * old_control - expected_control).mean()
    correction_loss = raw_correction if args.control_mode != "ppo" else pg_loss.new_zeros(())
    objective = pg_loss + correction_loss - args.ent_coef * entropy + args.vf_coef * value_loss + variance_loss
    with torch.no_grad():
        metrics = {
            "losses/policy_loss": pg_loss.detach(), "losses/value_loss": value_loss.detach(),
            "losses/entropy": entropy.detach(), "losses/old_approx_kl": (-logratio).mean(),
            "losses/approx_kl": ((ratio - 1) - logratio).mean(),
            "losses/clipfrac": ((ratio - 1).abs() > args.clip_coef).float().mean(),
            "control/train_loss": variance_loss.detach(), "control/correction_loss": correction_loss.detach(),
            "control/correction_mean": raw_correction.detach(),
            "control/analytic_response": (expected_control - old_coefficients[..., 0]).mean(),
        }
    return objective, metrics


@torch.no_grad()
def rollout_statistics(agent, observations, native_actions):
    """Detached, independently owned snapshots of every prefit behavior teacher."""
    alpha, beta, values = agent.get_policy_and_value(observations)
    moments = beta_log_moments(alpha, beta)
    score = torch.cat((native_actions.log(), torch.log1p(-native_actions)), dim=-1) - moments
    coefficients = agent.control(observations.detach())
    fisher_diagonal, fisher_cross = beta_fisher_components(alpha, beta)
    data = {
        "values": values.squeeze(-1), "logprobs": agent.action_logprob(alpha, beta, native_actions),
        "alpha": alpha, "beta": beta, "log_moments": moments, "score": score,
        "coefficients": coefficients, "gram": actor_jacobian_gram(agent.actor, observations),
        "fisher_diagonal": fisher_diagonal, "fisher_cross": fisher_cross,
        "logit_jacobian": beta_logit_jacobian(alpha, beta), "control_samples": control_sample(coefficients, score),
    }
    return {name: value.detach().clone() for name, value in data.items()}


@torch.no_grad()
def control_diagnostics(agent, observations, advantages, old_data, exact, norm_adv=True):
    """Prequential full-rollout diagnostics, evaluated BEFORE fitting.

    Full-rollout normalization is deterministic and differs from PPO's shuffled
    minibatch normalization; these are not post-clipping/Adam update variances.
    """
    advantages = advantages.detach()
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    coefficients, score = old_data["coefficients"], old_data["score"]
    samples = old_data["control_samples"]
    centered_control, centered_advantages = samples - samples.mean(), advantages - advantages.mean()
    control_variance = centered_control.square().mean()
    advantage_variance = centered_advantages.square().mean()
    metrics = {
        "control/preupdate_control_variance": control_variance,
        "control/preupdate_advantage_variance": advantage_variance,
        "control/preupdate_control_advantage_variance_ratio": control_variance / (advantage_variance + 1e-8),
        "control/preupdate_control_advantage_covariance": (centered_control * centered_advantages).mean(),
        "control/preupdate_correction_mean": (samples - coefficients[..., 0]).mean(),
    }
    if exact:
        jacobian = old_data["logit_jacobian"]
        base = jacobian * score * advantages.unsqueeze(-1)
        corrected = cached_corrected_output_gradient(
            advantages, score, coefficients, old_data["fisher_diagonal"], old_data["fisher_cross"], jacobian,
        )
        baseline = jacobian * score * (advantages - coefficients[..., 0]).unsqueeze(-1)
        base_second, base_mean_square = actor_gradient_moments(agent.actor, observations, base)
        corrected_second, corrected_mean_square = actor_gradient_moments(agent.actor, observations, corrected)
        baseline_second, baseline_mean_square = actor_gradient_moments(agent.actor, observations, baseline)
        _, shift_square = actor_gradient_moments(agent.actor, observations, corrected - base)
        base_variance = base_second - base_mean_square
        corrected_variance = corrected_second - corrected_mean_square
        baseline_variance = baseline_second - baseline_mean_square
        response = jacobian * cached_fisher_product(
            old_data["fisher_diagonal"], old_data["fisher_cross"], coefficients[..., 1:],
        )
        _, response_mean_square = actor_gradient_moments(agent.actor, observations, response)
        metrics.update({
            "control/preupdate_base_variance": base_variance,
            "control/preupdate_corrected_variance": corrected_variance,
            "control/preupdate_variance_ratio": corrected_variance / (base_variance + 1e-8),
            "control/preupdate_baseline_variance": baseline_variance,
            "control/preupdate_baseline_variance_ratio": baseline_variance / (base_variance + 1e-8),
            "control/preupdate_mean_gradient_shift_norm": shift_square.sqrt(),
            "control/preupdate_analytic_response_gradient_norm": response_mean_square.sqrt(),
            "control/preupdate_train_loss": 0.5 * corrected_second / agent.actor_parameter_count,
        })
    return {name: value.detach() for name, value in metrics.items()}


def optimizer_step(optimizers, parameters, max_grad_norm, norms):
    """Clip each exclusive owner separately after the one combined backward."""
    for index, group in enumerate(parameters):
        norms[index].copy_(nn.utils.clip_grad_norm_(group, max_grad_norm))
    for optimizer in optimizers:
        optimizer.step()


def validate_args(args):
    if args.env_id != "HalfCheetah-v4" or args.env_backend != "native":
        raise ValueError("v16 matched fixed-task runs require native HalfCheetah-v4")
    if not args.cuda or not args.compile:
        raise ValueError("v16 requires CUDA and compiled learner execution")
    if args.track:
        raise ValueError("v16 matched runs use local TensorBoard, not WandB")
    if args.control_mode not in {"ppo", "baseline", "corrected"}:
        raise ValueError("invalid control_mode")
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs, args.env_threads) <= 0:
        raise ValueError("environment, rollout, minibatch, epoch and thread counts must be positive")
    if not (0 < args.gamma < 1 and 0 <= args.gae_lambda <= 1):
        raise ValueError("gamma must be in (0,1), lambda in [0,1]")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not (0 < args.clip_coef < 1) or not np.isfinite(args.vf_coef) or args.vf_coef <= 0:
        raise ValueError("invalid clipping or critic coefficient")
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        raise ValueError("ent_coef must be finite and nonnegative")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_minibatches must divide batch_size exactly")
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size < 2 or args.total_timesteps < args.batch_size:
        raise ValueError("need at least two examples per minibatch and a complete rollout")
    return args


def make_training_env(args, run_name):
    return make_mujoco_vector_env(args.env_id, args.num_envs, backend=args.env_backend,
                                  num_threads=min(args.env_threads, args.num_envs),
                                  capture_video=False, run_name=run_name)


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
    with ExitStack() as resources:
        writer = SummaryWriter(f"runs/{run_name}")
        resources.callback(writer.close)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs, args).to(device)
        contract = {
            "version": 16, "control_mode": args.control_mode, "critic_mode": "scalar",
            "gamma": args.gamma, "gae_lambda": args.gae_lambda,
            "architecture": "separate_actor_critic_control_tanh64x64; native_host_actor; no_shared_trunk",
            "initialization": "canonical_v15_scalar_actor_critic_then_CPU_fork_rng_zero_control_head",
            "control_basis": "[1, log(u)-Eold_log(u), log(1-u)-Eold_log(1-u)]; alpha_all_then_beta_all",
            "control_units": "PPO_minibatch_normalized_advantage_if_norm_adv_else_raw_GAE; no_lagged_scaling",
            "training_objective": "0.5*mean(g^T K g)/actor_parameter_count; control_weight=1",
            "gradient": "g=J*(S*(A-c)+F*b); K=Jac_actor*Jac_actor^T_including_all_weights_and_biases",
            "conditional_normal_equation": "E[D^T*K*D|s]*b=E[D^T*K*J*S*A|s]; D=J*[S,SS^T-F]",
            "actor_correction": "mean(ratio*c_old-E_new[c_old]); baseline_arm_uses_only_b0; ppo_arm_uses_none",
            "control_target": "exact_same_detached_centered_std_normalized_advantage_as_PPO_minibatch",
            "optimizer_ownership": "exclusive_actor_critic_control_fused_Adam; identical_LR_annealing; separate_norm_clipping",
            "diagnostics": "preupdate_full_rollout_iterations1,16,32,...; full_rollout_advantage_normalization; "
                           "cheap_covariance_every_rollout; no_rng",
            "teacher_freezing": "alpha_beta_logmoments_scores_coefficients_samples_Gram_Fisher_J_values_logprobs_cached_before_any_fit",
            "optimum_scope": "exact_conditional_trace_for_ideal_Beta_unclipped_behavior_objective_unpreconditioned_actor_gradient",
            "not_claimed": "Adam_or_clipped_PPO_or_reused_batch_global_optimum; unbiased_GAE; generic_control_variate_novelty",
            "limitations": "epsilon_clipped_finite_precision_Beta_sampling_approximates_ideal_identity; "
                           "diagnostic_full_rollout_normalization_differs_from_PPO_minibatches; adaptive_PPO_reuse; "
                           "hidden_history_dependent_normalization; variance_diagnostics_are_prequential_not_future_guarantees",
        }
        writer.add_text("score_control_contract", str(contract))
        parameters = agent.parameter_groups()
        optimizers = tuple(optim.Adam(group, lr=args.learning_rate, eps=1e-5, fused=True) for group in parameters)
        counts = agent.parameter_counts()
        writer.add_text("parameter_counts", str(counts))
        print(f"parameter_counts={counts}")
        for name, count in counts.items():
            writer.add_scalar(f"parameters/{name}", count, 0)

        def statistics_model(observations, native):
            return rollout_statistics(agent, observations, native)

        def objective_model(observations, native, advantages, returns, old_data):
            return policy_loss(agent, observations, native, advantages, returns, old_data, args)

        def diagnostics_model(observations, advantages, old_data, exact):
            return control_diagnostics(agent, observations, advantages, old_data, exact, args.norm_adv)

        statistics_model = graph_compile(statistics_model)
        value_model = graph_compile(agent.get_value)
        diagnostics_model = graph_compile(diagnostics_model)
        objective_model = torch.compile(objective_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=True, mode=args.compile_mode, explicit_next_values=True)
        obs_shape = envs.single_observation_space.shape
        host_actor = HostPolicy(agent, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        fields = {"observations": obs_shape, "native_actions": (agent.action_dim,), "next_observations": obs_shape}
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers, fields=fields)
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        updates_per_rollout = args.update_epochs * args.num_minibatches
        gradient_norms = torch.empty((updates_per_rollout, 3), device=device)
        metric_names = ("losses/policy_loss", "losses/value_loss", "losses/entropy", "losses/old_approx_kl",
                        "losses/approx_kl", "losses/clipfrac", "control/train_loss", "control/correction_loss",
                        "control/correction_mean", "control/analytic_response")
        sums = {name: torch.zeros((), device=device) for name in metric_names}
        total_updates = 0
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon, phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                for optimizer in optimizers:
                    optimizer.param_groups[0]["lr"] = (1 - (iteration - 1) / args.num_iterations) * args.learning_rate
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, factual_next_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    # Factual terminal/time-limit observations; no reset bootstrap leakage.
                    transfer.push(step, reward, terms, truncs, observations=obs_step,
                                  native_actions=native, next_observations=factual_next_obs)
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
                # statistics_model has CUDA-graph reuse disabled and snapshots every
                # output, so subsequent optimizer updates cannot change these teachers.
                old_data = statistics_model(b_obs, b_native)
                next_values = value_model(batch.fields["next_observations"].flatten(0, 1)).reshape(
                    args.num_steps, args.num_envs,
                ).clone()
                advantages, returns = gae_fn(
                    batch.rewards, old_data["values"].view(args.num_steps, args.num_envs),
                    batch.terminations, batch.truncations, next_values, args.gamma, args.gae_lambda,
                )
                b_advantages, b_returns = advantages.flatten().detach().clone(), returns.flatten().detach().clone()
            with timer.span("diagnostics"):
                diagnostic_metrics = diagnostics_model(
                    b_obs, b_advantages, old_data, iteration == 1 or iteration % 16 == 0,
                )
            for accumulator in sums.values():
                accumulator.zero_()
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        torch.compiler.cudagraph_mark_step_begin()
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        objective, metrics = objective_model(
                            b_obs[indices], b_native[indices], b_advantages[indices], b_returns[indices],
                            {name: value[indices] for name, value in old_data.items()},
                        )
                        objective.backward()
                        optimizer_step(optimizers, parameters, args.max_grad_norm, gradient_norms[updates])
                        for name, metric in metrics.items():
                            sums[name].add_(metric.detach())
                        updates += 1
                host_actor.refresh()
            total_updates += updates
            metric_values = {name: value / updates for name, value in sums.items()}
            metric_values.update({
                "losses/explained_variance": explained_variance(old_data["values"], b_returns),
                "gradients/actor_norm": gradient_norms[:, 0].mean(),
                "gradients/critic_norm": gradient_norms[:, 1].mean(),
                "gradients/control_norm": gradient_norms[:, 2].mean(),
            })
            metric_values.update(diagnostic_metrics)
            logged = gather_metrics(metric_values)
            if any(not np.isfinite(value) for name, value in logged.items() if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            for name, value in {
                "ppo_steps": updates, "critic_steps": updates, "control_steps": updates,
                "ppo_steps_total": total_updates, "critic_steps_total": total_updates, "control_steps_total": total_updates,
                "ppo_examples": updates * args.minibatch_size, "critic_examples": updates * args.minibatch_size,
                "control_examples": updates * args.minibatch_size,
                "ppo_examples_total": total_updates * args.minibatch_size,
                "critic_examples_total": total_updates * args.minibatch_size,
                "control_examples_total": total_updates * args.minibatch_size,
                "ppo_minibatch_size": args.minibatch_size, "epochs": args.update_epochs,
            }.items():
                writer.add_scalar(f"updates/{name}", value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizers[0].param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (now - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save({
                "model": agent.state_dict(), "args": vars(args), "global_step": global_step,
                "parameter_counts": counts, "control_mode": args.control_mode,
                "score_control_contract": contract,
                "optimizer_steps": total_updates,
                "optimizers": {name: optimizer.state_dict()
                               for name, optimizer in zip(("actor", "critic", "control"), optimizers)},
                "reward_norm": {
                    "returns": torch.from_numpy(rew_norm.returns.copy()),
                    "means": torch.from_numpy(rew_norm.means.copy()),
                    "variances": torch.from_numpy(rew_norm.variances.copy()),
                    "counts": torch.from_numpy(rew_norm.counts.copy()),
                    "gamma": rew_norm.gamma, "epsilon": rew_norm.epsilon, "clip": rew_norm.clip,
                },
                "obs_norm": {
                    "means": torch.from_numpy(obs_norm.means.copy()),
                    "variances": torch.from_numpy(obs_norm.variances.copy()),
                    "counts": torch.from_numpy(obs_norm.counts.copy()),
                    "epsilon": obs_norm.epsilon, "clip": obs_norm.clip,
                },
            }, model_path)
            print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()
