# State-only vector successor features with matched auxiliary/consumer critics.
# psi=(1-gamma) E_pi[sum_k gamma**k phi_{t+k}], where phi starts with RAW
# observed reward. No reward reconstruction fit or online feature standardizer.
# Reward-only supervision is an auxiliary scalar value prediction, NOT evidence
# for rich vector targets. Joint/reward contrasts target richness; critic/auxiliary
# contrasts consumption with identical critic capacity (not v6 critic capacity).
# Both use frozen rollout critic inputs. Between-rollout encoder/predictor drift
# remains; forecasts are policy-conditional, not policy-independent dynamics.
# PPO reads only detached CURRENT-state predictions, never future observations,
# actions or rewards. Its nonlinear critic does not assert V = w.psi.
# jepa_mode selects only the critic BASE (raw vs latent); consumed forecasts
# remain JEPA-derived even in actor mode. PPO stops at encoder and SF head;
# representation owns the encoder, including optional attached SF gradients.
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

from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.lejepa import ActionEncoder, FeedForward, MLP, SIGReg
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

    jepa_mode: Literal["actor", "both"] = "both"
    """critic BASE: actor=raw, both=latent; consumed forecasts remain JEPA-derived in either mode"""
    sigreg_weight: float = 0.09
    """LeWM coefficient on the raw, batch-size-scaled Epps-Pulley statistic"""
    sigreg_num_proj: int = 1024
    """fresh random projection directions per SIGReg call, matching le-wm"""
    sigreg_proj_chunk: int = 256
    """directions per memory-bounded chunk; does not change the statistic"""
    ssl_learning_rate: float = 5e-5
    """constant representation AdamW learning rate, matching le-wm"""
    ssl_weight_decay: float = 1e-3
    """representation AdamW weight decay, matching le-wm"""
    balance_interval: int = 10
    """sample prediction/SIGReg/reward encoder gradients every N rollouts; 0 disables"""

    reward_mode: Literal["off", "detached", "attached"] = "detached"
    """one-step normalized reward supervision; only attached trains the encoder"""
    reward_coef: float = 1.0
    """fixed reward MSE coefficient; no loss normalization"""
    task_activation: Literal["tanh", "stiglu"] = "tanh"
    """task FFNs only; SiTU-GLU is not parameter-matched to Tanh"""
    task_residual: bool = False
    """add one parameter-free identity skip around the second task stage"""
    weight_projection: Literal["none", "hidden", "all"] = "none"
    """unit task weight vectors; all also replaces the 0.01 actor-head scale"""
    ssl_minibatch_size: int = 512
    """fixed SSL batch size independent of PPO minibatch size"""

    successor_targets: Literal["reward", "joint"] = "joint"
    """same D outputs; reward trains/consumes only raw-reward channel zero"""
    successor_consumer: Literal["auxiliary", "critic"] = "critic"
    """independent forecast path: matched critic zero slots vs detached JEPA-derived psi"""
    successor_encoder_gradients: Literal["stopped", "attached"] = "stopped"
    """SF loss current encoder gradients only; teacher/critic always detached"""

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


class ActionConditionedMLP(nn.Module):
    """LeWM's AdaLN-zero feed-forward path, without attention or temporal tokens."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(64, 64)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(64, 3 * 64))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.output_norm = nn.LayerNorm(64)

    def forward(self, latent, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        hidden = self.norm(latent) * (1.0 + scale) + shift
        return self.output_norm(latent + gate * self.mlp(hidden))


class LeWMBranch(nn.Module):
    """Training-only LeWM projection/prediction stack on the shared encoder.

    Use the repo's LayerNorm MLP port (not running-stat BatchNorm) for CUDA-graph
    safety. Its output is unbounded: SIGReg must not test bounded tanh features.
    The statistic sees (time=2, batch, latent), testing each temporal marginal.
    """

    def __init__(self, action_dim, num_proj, proj_chunk):
        super().__init__()
        self.projector = MLP(64, 64, 64)
        self.action_encoder = ActionEncoder(action_dim, 64)
        self.predictor = ActionConditionedMLP()
        self.pred_proj = MLP(64, 64, 64)
        self.sigreg = SIGReg(knots=17, num_proj=num_proj, proj_chunk=proj_chunk)

    def forward(self, current, following, native_actions):
        # A single shared projector encodes both attached temporal branches.
        embeddings = self.projector(torch.stack((current, following)))
        prediction = self.pred_proj(self.predictor(
            embeddings[0], self.action_encoder(2.0 * native_actions - 1.0),
        ))
        prediction_loss = F.mse_loss(prediction, embeddings[1])
        regularization = self.sigreg(embeddings)
        with torch.no_grad():
            projected_std = embeddings.std(dim=1, correction=0).mean()
            backbone_std = following.std(dim=0, correction=0).mean()
        return prediction_loss, regularization, projected_std, backbone_std


class TaskFFN(nn.Module):
    """Two width-64 stages and a linear head; residual adds no parameters."""

    def __init__(self, input_dim, output_dim, output_std, activation="tanh", residual=False):
        super().__init__()
        if activation not in {"tanh", "stiglu"}:
            raise ValueError("invalid task activation")
        self.residual = residual

        def stage(width):
            if activation == "stiglu":
                return nn.Sequential(init_situglu_branch(SiTUGLUBranch(width, 64)))
            return nn.Sequential(layer_init(nn.Linear(width, 64)), nn.Tanh())

        # Keep Linear construction/initialization interleaved exactly as v4.
        self.first = stage(input_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))

    def forward(self, x):
        hidden = self.first(x)
        branch = self.second(hidden)
        return self.head(hidden + branch if self.residual else branch)


class HostPolicy:
    """Native FP32 rollout mirror; no auxiliary training modules.

    Nonresidual arms keep v4's single fused native graph. Residual arms use
    permanent stage buffers and one identity addition, with live weight sources.
    """

    def __init__(self, agent, num_rows):
        self.residual = agent.actor.residual
        if not self.residual:
            layers = list(agent.encoder) if agent.actor_uses_encoder else []
            layers.extend(agent.actor.first)
            layers.extend(agent.actor.second)
            layers.extend(agent.actor.head)
            self.fused = make_host_mirror(nn.Sequential(*layers), num_rows)
        else:
            self.encoder = make_host_mirror(agent.encoder, num_rows) if agent.actor_uses_encoder else None
            self.first = make_host_mirror(agent.actor.first, num_rows)
            self.second = make_host_mirror(agent.actor.second, num_rows)
            self.head = make_host_mirror(agent.actor.head, num_rows)
            self.hidden = np.empty((num_rows, 64), dtype=np.float32)

    def refresh(self):
        if not self.residual:
            self.fused.refresh()
        else:
            if self.encoder is not None:
                self.encoder.refresh()
            self.first.refresh()
            self.second.refresh()
            self.head.refresh()

    def __call__(self, observations):
        if not self.residual:
            return self.fused(observations)
        features = self.encoder(observations) if self.encoder is not None else observations
        np.copyto(self.hidden, self.first(features))
        np.add(self.hidden, self.second(self.hidden), out=self.hidden)
        return self.head(self.hidden)


class Agent(nn.Module):
    def __init__(self, envs, args):
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
        if args.jepa_mode not in {"actor", "both"}:
            raise ValueError("v8 supports actor/both JEPA modes only")
        if args.successor_targets not in {"reward", "joint"}:
            raise ValueError("invalid successor target channels")
        if args.successor_consumer not in {"auxiliary", "critic"}:
            raise ValueError("invalid successor consumer")
        if args.successor_encoder_gradients not in {"stopped", "attached"}:
            raise ValueError("invalid successor encoder gradient path")
        if args.reward_mode not in {"off", "detached", "attached"}:
            raise ValueError("invalid reward mode")
        if args.task_activation not in {"tanh", "stiglu"}:
            raise ValueError("invalid task activation")
        if args.weight_projection not in {"none", "hidden", "all"}:
            raise ValueError("invalid weight projection")
        if min(args.sigreg_num_proj, args.sigreg_proj_chunk) <= 0:
            raise ValueError("SIGReg projection counts must be positive")
        self.actor_uses_encoder = True
        self.critic_uses_encoder = args.jepa_mode == "both"
        self.reward_mode = args.reward_mode
        self.weight_projection = args.weight_projection
        self.observation_dim = observation_dim
        self.successor_dim = 1 + observation_dim + 2 * self.action_dim + 1
        self.successor_targets = args.successor_targets
        self.successor_consumer = args.successor_consumer
        self.successor_encoder_gradients = args.successor_encoder_gradients
        self.active_successor_channels = self.successor_dim if args.successor_targets == "joint" else 1
        mask = torch.ones(self.successor_dim)
        if args.successor_targets == "reward":
            mask[1:] = 0.0
        self.register_buffer("successor_mask", mask)
        critic_input = 64 if self.critic_uses_encoder else observation_dim
        # Always consume canonical v6 BOTH RNG: critic64 -> actor64 -> encoder
        # -> JEPA. Actor vs both must not shift representation/policy starting
        # tensors just because the raw-observation critic has a different width.
        self.critic = TaskFFN(64, 1, 1.0, "tanh", args.task_residual)
        self.actor = TaskFFN(64, 2 * self.action_dim, 0.01, "tanh", args.task_residual)
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.ssl = LeWMBranch(self.action_dim, args.sigreg_num_proj, args.sigreg_proj_chunk)
        if args.task_activation == "stiglu":
            with torch.random.fork_rng(devices=[]):
                # Preserve v6 both's GLU actor initialization too.
                self.critic = TaskFFN(64, 1, 1.0, "stiglu", args.task_residual)
                self.actor = TaskFFN(64, 2 * self.action_dim, 0.01, "stiglu", args.task_residual)
        with torch.random.fork_rng(devices=[]):
            self.reward_head = nn.Sequential(
                layer_init(nn.Linear(64 + self.action_dim, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
        # Independent RNG forks: no treatment can advance the canonical stream.
        # Every consumer/target arm has exactly the same trainable architecture.
        with torch.random.fork_rng(devices=[]):
            self.critic = TaskFFN(critic_input + self.successor_dim, 1, 1.0,
                                  args.task_activation, args.task_residual)
        with torch.random.fork_rng(devices=[]):
            self.successor_head = nn.Sequential(
                layer_init(nn.Linear(64, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, self.successor_dim), std=0.01),
            )
        self.project_policy_weights()

    def parameter_groups(self):
        """Disjoint PPO, representation, reward-head, and successor-head owners."""
        policy = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
        representation = tuple(self.encoder.parameters()) + tuple(self.ssl.parameters())
        return policy, representation, tuple(self.reward_head.parameters()), tuple(self.successor_head.parameters())

    def parameter_counts(self):
        counts = {
            "shared_encoder": 0 if self.encoder is None else sum(p.numel() for p in self.encoder.parameters()),
            "actor_ffn": sum(p.numel() for p in self.actor.parameters()),
            "critic_ffn": sum(p.numel() for p in self.critic.parameters()),
            "jepa_auxiliary": 0 if self.ssl is None else sum(p.numel() for p in self.ssl.parameters()),
            "reward_head": sum(p.numel() for p in self.reward_head.parameters()),
            "successor_head": sum(p.numel() for p in self.successor_head.parameters()),
        }
        counts["inference"] = counts["shared_encoder"] + counts["actor_ffn"] + counts["critic_ffn"]
        counts["training_only"] = counts["jepa_auxiliary"] + counts["reward_head"]
        counts["inference" if self.successor_consumer == "critic" else "training_only"] += counts["successor_head"]
        counts["total"] = counts["inference"] + counts["training_only"]
        return counts

    @torch.no_grad()
    def project_policy_weights(self):
        """Unit task weight vectors only; never modify biases or Adam moments."""
        if self.weight_projection == "none":
            return
        for task in (self.actor, self.critic):
            for stage in (task.first, task.second):
                branch = stage[0]
                axes = (((branch.gate.weight, 1), (branch.up.weight, 1), (branch.down.weight, 0))
                        if isinstance(branch, SiTUGLUBranch) else ((branch.weight, 1),))
                for weight, dim in axes:
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
            if self.weight_projection == "all":
                weight = task.head[0].weight
                weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def reward_prediction(self, latent, native_actions):
        """Predict stored one-step reward, using the centered action actually taken."""
        features = latent if self.reward_mode == "attached" else latent.detach()
        return self.reward_head(torch.cat((features, 2.0 * native_actions.detach() - 1.0), dim=-1)).flatten()

    def successor_prediction(self, latent):
        """State-only forecast. This flag controls ONLY the current SF student."""
        features = latent if self.successor_encoder_gradients == "attached" else latent.detach()
        return self.successor_head(features)

    def get_successor(self, x):
        return self.successor_prediction(self.encoder(x))

    def critic_features(self, x, latent, successors=None):
        """PPO must never train either the encoder or successor predictor.

        Rollout snapshots call this once and retain the whole base+D tensor for
        every PPO epoch. At inference, the current state alone generates slots.
        """
        base = latent.detach() if self.critic_uses_encoder else x.detach()
        if self.successor_consumer == "auxiliary":
            slots = base.new_zeros((*base.shape[:-1], self.successor_dim))
        else:
            if successors is None:
                successors = self.successor_head(latent.detach())
            slots = successors.detach() * self.successor_mask
        return torch.cat((base, slots), dim=-1)

    def get_value(self, x):
        latent = self.encoder(x)
        return self.critic(self.critic_features(x, latent))

    def get_policy_and_value(self, x, *, critic_inputs=None):
        alpha, beta, value, _ = self.get_policy_value_latents(x, critic_inputs=critic_inputs)
        return alpha, beta, value

    def get_policy_value_latents(self, x, *, critic_inputs=None):
        # Actor/SSL are online; critic PPO coordinates are frozen for the fit.
        latent = self.encoder(x)
        logits = self.actor(latent.detach())
        inputs = self.critic_features(x, latent) if critic_inputs is None else critic_inputs.detach()
        value = self.critic(inputs)
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        return alpha, beta, value, latent

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


def policy_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args,
                *, critic_inputs=None):
    """Unchanged clipped Beta PPO; shared encoder inputs are always detached."""
    return _policy_objective(agent, agent.get_policy_and_value(observations, critic_inputs=critic_inputs), native_actions,
                             old_logprobs, advantages, returns, old_values, args)


def _policy_objective(agent, outputs, native_actions, old_logprobs, advantages, returns, old_values, args):
    alpha, beta, newvalue = outputs
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
    actor_objective = pg_loss - args.ent_coef * entropy_loss
    critic_objective = v_loss * args.vf_coef
    metrics = {
        "losses/policy_loss": pg_loss.detach(),
        "losses/value_loss": v_loss.detach(),
        "losses/entropy": entropy_loss.detach(),
        "losses/old_approx_kl": old_approx_kl,
        "losses/approx_kl": approx_kl,
        "losses/clipfrac": clipfrac,
    }
    return actor_objective + critic_objective, metrics


@torch.no_grad()
def successor_features(raw_rewards, observations, native_actions):
    """Fixed per-transition phi in physical task units, not PPO reward units.

    Observation rows are the stored normalized rollout observations; actions are
    centered native samples (2*native-1). No adaptive statistics reparameterize
    a bootstrap, and no approximate reward reconstruction is necessary.
    """
    centered = 2.0 * native_actions - 1.0
    return torch.cat((raw_rewards.unsqueeze(-1), observations, centered,
                      centered.square(), torch.ones_like(raw_rewards).unsqueeze(-1)), dim=-1)


@torch.no_grad()
def successor_lambda_targets(features, predictions, next_predictions, terminations, truncations,
                             gamma, gae_lambda, gae_fn):
    """Frozen vector TD(lambda), using the shared bounded-code GAE primitive.

    Inputs have (T,N,D) shape; next_predictions MUST be on factual transition
    observations, not reset observations. Term (including term+trunc) cuts both
    bootstrap and trace. Trunc alone keeps factual bootstrap but cuts trace.
    The last rollout row has one-step bootstrap and no unavailable future trace.
    """
    steps, environments, channels = features.shape
    terms = terminations.unsqueeze(-1).expand(steps, environments, channels).reshape(steps, -1)
    truncs = truncations.unsqueeze(-1).expand(steps, environments, channels).reshape(steps, -1)
    _, targets = gae_fn(
        ((1.0 - gamma) * features.detach()).reshape(steps, -1),
        predictions.detach().reshape(steps, -1), terms, truncs,
        next_predictions.detach().reshape(steps, -1), gamma, gae_lambda,
    )
    # Own storage: subsequent compiled GAE calls must not overwrite the teacher.
    return targets.reshape_as(features).detach().clone()


def successor_objective(agent, predictions, targets):
    """One fixed mean MSE over active channels, in normalized occupancy units."""
    active_error = (predictions - targets.detach())[..., :agent.active_successor_channels]
    return active_error.square().mean()


@torch.no_grad()
def successor_error_metrics(agent, predictions, targets):
    """Unstandardized occupancy errors per feature block (inactive blocks labeled).

    Reward occupancy divided by (1-gamma) is a scalar raw-reward return forecast;
    it is not itself the nonlinear PPO critic, nor evidence for vector benefits.
    """
    errors = (predictions - targets).square().mean(dim=0)
    obs_end = 1 + agent.observation_dim
    action_end = obs_end + agent.action_dim
    prefix = "successor" if agent.successor_targets == "joint" else "successor/inactive"
    return {
        "successor/vector_mse": successor_objective(agent, predictions, targets),
        "successor/raw_reward_occupancy_mse": errors[0],
        f"{prefix}/raw_observation_occupancy_mse": errors[1:obs_end].mean(),
        f"{prefix}/raw_action_occupancy_mse": errors[obs_end:action_end].mean(),
        f"{prefix}/raw_squared_action_occupancy_mse": errors[action_end:-1].mean(),
        f"{prefix}/raw_survival_occupancy_mse": errors[-1],
    }


@torch.no_grad()
def successor_fit_diagnostics(agent, observations, cached_inputs, cached_predictions, targets, gamma):
    """Same CURRENT critic head on old vs new inputs isolates coordinate drift.

    Coordinates cannot move within the PPO fit; these diagnostics expose the
    remaining inference discontinuity between rollouts, after all learner updates.
    """
    latent = agent.encoder(observations)
    predictions = agent.successor_head(latent)
    current_inputs = agent.critic_features(observations, latent, predictions)
    old_value = agent.critic(cached_inputs)
    new_value = agent.critic(current_inputs)
    base_width = cached_inputs.shape[-1] - agent.successor_dim
    metrics = {
        "successor/prefit_vector_mse": successor_objective(agent, cached_predictions, targets),
        "successor/postfit_vector_mse": successor_objective(agent, predictions, targets),
        "successor/prefit_raw_reward_return_mse":
            ((cached_predictions[:, 0] - targets[:, 0]) / (1.0 - gamma)).square().mean(),
        "successor/postfit_raw_reward_return_mse":
            ((predictions[:, 0] - targets[:, 0]) / (1.0 - gamma)).square().mean(),
        "drift/current_critic_value_rmse": (new_value - old_value).square().mean().sqrt(),
        "drift/critic_base_input_rmse":
            (current_inputs[:, :base_width] - cached_inputs[:, :base_width]).square().mean().sqrt(),
        "drift/critic_successor_input_rmse":
            (current_inputs[:, base_width:] - cached_inputs[:, base_width:]).square().mean().sqrt(),
        "drift/successor_prediction_rmse":
            (predictions - cached_predictions).square().mean().sqrt(),
    }
    metrics.update({name.replace("successor/", "successor/postfit/", 1): value
                    for name, value in successor_error_metrics(agent, predictions, targets).items()})
    return metrics


def representation_components(agent, observations, native_actions, next_observations, rewards,
                              successor_targets, args, *, latent=None):
    """Weighted prediction/SIGReg/reward components from ONE stochastic forward.

    The reward target is RolloutTransfer's real normalized one-step reward, not
    the PPO return or a bootstrapped target. Neither actions nor targets carry
    gradients. Prediction and SIGReg retain exactly v4's unnormalized statistic.
    """
    latent = agent.encoder(observations) if latent is None else latent
    prediction, regularization, projected_std, backbone_std = agent.ssl(
        latent, agent.encoder(next_observations), native_actions.detach(),
    )
    metrics = {
        "ssl/shared_prediction_loss": prediction.detach(),
        "ssl/shared_sigreg_loss": regularization.detach(),
        "ssl/shared_projected_std": projected_std,
        "ssl/shared_backbone_std": backbone_std,
    }
    reward_objective = observations.new_zeros(())
    if agent.reward_mode != "off":
        predicted_reward = agent.reward_prediction(latent, native_actions)
        target = rewards.detach().reshape(-1)
        reward_mse = F.mse_loss(predicted_reward, target)
        reward_objective = args.reward_coef * reward_mse
        metrics["reward/mse"] = reward_mse.detach()
        with torch.no_grad():
            metrics["reward/explained_variance"] = explained_variance(predicted_reward, target)
    successors = agent.successor_prediction(latent)
    successor_loss = successor_objective(agent, successors, successor_targets)
    metrics.update(successor_error_metrics(agent, successors.detach(), successor_targets.detach()))
    return torch.stack((prediction, args.sigreg_weight * regularization,
                        reward_objective, successor_loss)), metrics


def representation_loss(agent, observations, native_actions, next_observations, rewards, successor_targets, args):
    components, metrics = representation_components(
        agent, observations, native_actions, next_observations, rewards, successor_targets, args,
    )
    return components[:2].sum(), components[2], components[3], metrics


def gradient_balance(agent, components):
    """Weighted encoder gradients without modifying any parameter's .grad.

    Reuse these very components for backward afterwards: diagnostics must not
    introduce additional SIGReg batches or advance its random stream. A detached
    reward objective has exactly zero encoder gradient, not a missing metric.
    """
    if agent.encoder is None:
        return {}
    parameters = tuple(agent.encoder.parameters())
    gradients = [torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
                 if component.requires_grad else (None,) * len(parameters)
                 for component in components.unbind()]
    with torch.no_grad():
        zero = components.new_zeros(())
        norms = [sum((grad.square().sum() for grad in row if grad is not None), zero).sqrt()
                 for row in gradients]

        def cosine(left, right):
            dot = sum(((a * b).sum() for a, b in zip(gradients[left], gradients[right])
                       if a is not None and b is not None), zero)
            return dot / (norms[left] * norms[right]).clamp_min(1e-12)

        return {
            "balance/shared_prediction_grad_norm": norms[0],
            "balance/shared_sigreg_grad_norm": norms[1],
            "balance/shared_reward_grad_norm": norms[2],
            "balance/shared_sigreg_to_prediction_grad_ratio": norms[1] / norms[0].clamp_min(1e-12),
            "balance/shared_reward_to_prediction_grad_ratio": norms[2] / norms[0].clamp_min(1e-12),
            "balance/shared_grad_cosine": cosine(0, 1),
            "balance/shared_prediction_reward_grad_cosine": cosine(0, 2),
            "balance/shared_sigreg_reward_grad_cosine": cosine(1, 2),
            "balance/shared_successor_grad_norm": norms[3],
            "balance/shared_successor_to_prediction_grad_ratio": norms[3] / norms[0].clamp_min(1e-12),
            "balance/shared_prediction_successor_grad_cosine": cosine(0, 3),
            "balance/shared_sigreg_successor_grad_cosine": cosine(1, 3),
        }


def iter_update_batches(batch_size, ppo_size, ssl_size, device, generator=None):
    """One permutation per epoch; every example has exactly one SSL exposure.

    Yield the whole PPO row only alongside its first SSL subrow. Subsequent
    subrows update world/reward owners only. Thus PPO update counts may change
    while SSL batch size, examples and optimizer steps remain fixed.
    """
    if min(batch_size, ppo_size, ssl_size) <= 0:
        raise ValueError("batch and minibatch sizes must be positive")
    if batch_size % ppo_size or ppo_size < ssl_size or ppo_size % ssl_size:
        raise ValueError("PPO batches must divide the rollout and contain whole SSL batches")
    for indices in device_minibatches(batch_size, ppo_size, device, generator):
        for offset in range(0, ppo_size, ssl_size):
            yield (indices if offset == 0 else None), indices[offset:offset + ssl_size]


def optimizer_step(agent, optimizers, args, *, policy_step=True, parameters=None, norms=None, projection=None):
    """Clip each owner separately; project task matrices after PPO steps only.

    The main loop passes cached parameter tuples and a preallocated norm row.
    This helper deliberately does not zero gradients: it consumes one combined
    backward before any owner steps, including on a PPO row's first SSL subrow.
    """
    parameters = agent.parameter_groups() if parameters is None else parameters
    if norms is None:
        norms = agent.action_low.new_zeros(4)
    else:
        norms.zero_()
    for index, (optimizer, group) in enumerate(zip(optimizers, parameters)):
        if optimizer is not None and (index != 0 or policy_step):
            norms[index].copy_(nn.utils.clip_grad_norm_(group, 0.5 if index == 3 else args.max_grad_norm))
    for index, optimizer in enumerate(optimizers):
        if optimizer is not None and (index != 0 or policy_step):
            optimizer.step()
            if index == 0:
                (agent.project_policy_weights if projection is None else projection)()
    return norms


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    if args.jepa_mode not in {"actor", "both"}:
        raise ValueError("v8 supports actor/both JEPA modes only")
    if args.successor_targets not in {"reward", "joint"}:
        raise ValueError("invalid successor target channels")
    if args.successor_consumer not in {"auxiliary", "critic"}:
        raise ValueError("invalid successor consumer")
    if args.successor_encoder_gradients not in {"stopped", "attached"}:
        raise ValueError("invalid successor encoder gradient path")
    if not np.isfinite(args.gamma) or not 0 <= args.gamma < 1:
        raise ValueError("normalized successor occupancy requires 0 <= gamma < 1")
    if not np.isfinite(args.gae_lambda) or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gae_lambda must lie in [0,1]")
    if args.reward_mode not in {"off", "detached", "attached"}:
        raise ValueError("invalid reward mode")
    if args.task_activation not in {"tanh", "stiglu"}:
        raise ValueError("invalid task activation")
    if args.weight_projection not in {"none", "hidden", "all"}:
        raise ValueError("invalid weight projection")
    if not np.isfinite(args.reward_coef) or args.reward_coef != 1.0:
        raise ValueError("reward_coef is fixed at 1.0 in this ablation")
    if not np.isfinite(args.sigreg_weight) or args.sigreg_weight < 0:
        raise ValueError("sigreg_weight must be finite and nonnegative")
    if min(args.sigreg_num_proj, args.sigreg_proj_chunk) <= 0:
        raise ValueError("SIGReg projection counts must be positive")
    if not np.isfinite(args.ssl_learning_rate) or args.ssl_learning_rate <= 0:
        raise ValueError("ssl_learning_rate must be finite and positive")
    if not np.isfinite(args.ssl_weight_decay) or args.ssl_weight_decay < 0:
        raise ValueError("ssl_weight_decay must be finite and nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if args.balance_interval < 0:
        raise ValueError("balance_interval must be nonnegative")
    if args.ssl_minibatch_size != 512:
        raise ValueError("ssl_minibatch_size must remain 512 to preserve the SIGReg statistic")
    if args.target_kl is not None:
        raise ValueError("target_kl is unsupported: early stopping would change fixed SSL exposure")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_minibatches must divide batch_size exactly")
    args.minibatch_size = args.batch_size // args.num_minibatches
    if (args.minibatch_size < args.ssl_minibatch_size
            or args.minibatch_size % args.ssl_minibatch_size):
        raise ValueError("PPO minibatch size must be at least SSL512 and a multiple of it")
    if args.total_timesteps < args.batch_size:
        raise ValueError("total_timesteps must cover at least a full rollout")
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
        agent = Agent(envs, args).to(device)
        parameters = agent.parameter_groups()
        policy_parameters, ssl_parameters, reward_parameters, successor_parameters = parameters
        optimizer = optim.Adam(policy_parameters, lr=args.learning_rate, eps=1e-5, fused=True)
        ssl_optimizer = optim.AdamW(ssl_parameters, lr=args.ssl_learning_rate,
                                   weight_decay=args.ssl_weight_decay, fused=True) if ssl_parameters else None
        reward_optimizer = (optim.AdamW(reward_parameters, lr=args.ssl_learning_rate,
                                       weight_decay=args.ssl_weight_decay, fused=True)
                            if reward_parameters and args.reward_mode != "off" else None)
        successor_optimizer = optim.AdamW(successor_parameters, lr=args.ssl_learning_rate,
                                          weight_decay=args.ssl_weight_decay, fused=True)
        optimizers = (optimizer, ssl_optimizer, reward_optimizer, successor_optimizer)
        project_weights = agent.project_policy_weights
        if args.compile and args.weight_projection != "none":
            project_weights = torch.compile(
                project_weights, fullgraph=True, options={"triton.cudagraphs": False},
            )
        counts = agent.parameter_counts()
        writer.add_text("parameter_counts", str(counts))
        print(f"parameter_counts={counts}")
        for name, count in counts.items():
            writer.add_scalar(f"parameters/{name}", count, 0)
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """One preupdate snapshot supplies values, SF teachers and critic inputs."""
            latent = agent.encoder(observations)
            successors = agent.successor_head(latent)
            critic_inputs = agent.critic_features(observations, latent, successors)
            value = agent.critic(critic_inputs)
            alpha, beta = (F.softplus(agent.actor(latent)) + 1.0).chunk(2, dim=-1)
            return value.flatten(), agent.action_logprob(alpha, beta, native), successors, critic_inputs

        successor_model = agent.get_successor

        def joint_model(observations, native, old_logprobs, advantages, returns, old_values,
                        next_observations, rewards, successor_targets, critic_inputs):
            alpha, beta, value, latent = agent.get_policy_value_latents(
                observations, critic_inputs=critic_inputs,
            )
            ppo, ppo_metrics = _policy_objective(
                agent, (alpha, beta, value), native, old_logprobs, advantages, returns, old_values, args,
            )
            # The first SSL subrow is a prefix of this PPO row. Reuse the current
            # latent: at PPO512 this is exactly v4's single encoder evaluation.
            components, world_metrics = representation_components(
                agent, observations[:args.ssl_minibatch_size], native[:args.ssl_minibatch_size],
                next_observations, rewards, successor_targets, args,
                latent=latent[:args.ssl_minibatch_size],
            )
            return ppo, components, ppo_metrics, world_metrics

        def world_model(observations, native, next_observations, rewards, successor_targets):
            return representation_components(agent, observations, native, next_observations,
                                             rewards, successor_targets, args)

        def fit_diagnostics_model(observations, cached_inputs, cached_predictions, targets):
            return successor_fit_diagnostics(
                agent, observations, cached_inputs, cached_predictions, targets, args.gamma,
            )

        diagnostic_model = joint_model
        if args.compile:
            # Repeated objective gradients followed by backward must not alias
            # CUDA-graph output buffers; ordinary updates retain CUDA graphs.
            diagnostic_model = torch.compile(joint_model, fullgraph=True, dynamic=False,
                                             options={"triton.cudagraphs": False})
            joint_model = torch.compile(joint_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            world_model = torch.compile(world_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
            rollout_statistics = graph_compile(rollout_statistics)
            successor_model = graph_compile(successor_model)
            fit_diagnostics_model = graph_compile(fit_diagnostics_model)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        successor_gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode,
                                    explicit_next_values=True)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = HostPolicy(agent, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        fields = {"observations": obs_shape, "native_actions": (agent.action_dim,),
                  "next_observations": obs_shape, "raw_rewards": ()}
        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields=fields)
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * (args.batch_size // args.ssl_minibatch_size)
        gradient_norms = torch.empty((max_updates, 4), device=device)
        ppo_metric_names = ("losses/policy_loss", "losses/value_loss", "losses/entropy",
                            "losses/old_approx_kl", "losses/approx_kl", "losses/clipfrac")
        successor_prefix = "successor" if args.successor_targets == "joint" else "successor/inactive"
        world_metric_names = (
            "ssl/shared_prediction_loss", "ssl/shared_sigreg_loss",
            "ssl/shared_projected_std", "ssl/shared_backbone_std",
            "successor/vector_mse", "successor/raw_reward_occupancy_mse",
            f"{successor_prefix}/raw_observation_occupancy_mse",
            f"{successor_prefix}/raw_action_occupancy_mse",
            f"{successor_prefix}/raw_squared_action_occupancy_mse",
            f"{successor_prefix}/raw_survival_occupancy_mse",
        )
        if reward_optimizer is not None:
            world_metric_names += ("reward/mse", "reward/explained_variance")
        # Each metric is accumulated detached on device, once per matching owner
        # update. PPO and SSL denominators are deliberately independent.
        ppo_sums = {name: torch.zeros((), device=device) for name in ppo_metric_names}
        world_sums = {name: torch.zeros((), device=device) for name in world_metric_names}
        total_ppo_updates = total_ssl_updates = total_reward_updates = total_successor_updates = 0
        writer.add_text("ablation", f"{args.task_activation=}, {args.task_residual=}, "
                        f"{args.weight_projection=}, {args.reward_mode=}; fixed SSL512; "
                        "nGPT-style weight-only projection, not full nGPT; SiTU-GLU changes capacity")
        writer.add_text("successor_contract", f"{args.successor_targets=}, {args.successor_consumer=}, "
                        f"{args.successor_encoder_gradients=}; phi=[raw reward, normalized obs, "
                        "2*native-1, (2*native-1)^2, 1]; normalized discounted occupancy; "
                        "fixed preupdate factual TD(lambda) teachers; fixed base+D critic inputs "
                        "through all epochs; between-rollout encoder/SF drift remains. "
                        f"{args.jepa_mode=} selects raw/latent critic BASE only; actor always "
                        "consumes JEPA latents, and consumed SF is JEPA-derived in either mode. "
                        "PPO stops at encoder and SF head; representation optimizer owns encoder, "
                        "including optional attached current-student SF gradients; teachers detached. "
                        "Auxiliary uses zero slots with matched critic capacity; reward-only is "
                        "scalar value supervision, not vector evidence. Current-state inference "
                        "only; predictions remain policy-conditional; nonlinear critic, not V=w.psi.")
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
                    # Includes physical terminal/time-limit observations, never reset states.
                    # PPO and the one-step reward head still use normalized `reward`.
                    transfer.push(step, reward, terms, truncs, observations=obs_step,
                                  native_actions=native, next_observations=transition_obs,
                                  raw_rewards=raw_reward)
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
                b_next_obs = batch.fields["next_observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_rewards = batch.rewards.flatten()
                snapshot_values, snapshot_logprobs, snapshot_successors, snapshot_inputs = rollout_statistics(b_obs, b_native)
                # These coordinates and teachers outlive all compiled update calls.
                b_values, b_logprobs = snapshot_values.clone(), snapshot_logprobs.clone()
                b_successors, b_critic_inputs = snapshot_successors.clone(), snapshot_inputs.clone()
                next_successors = successor_model(b_next_obs)
                features = successor_features(
                    batch.fields["raw_rewards"], batch.fields["observations"],
                    batch.fields["native_actions"],
                )
                b_successor_targets = successor_lambda_targets(
                    features, b_successors.view(args.num_steps, args.num_envs, agent.successor_dim),
                    next_successors.view(args.num_steps, args.num_envs, agent.successor_dim),
                    batch.terminations, batch.truncations, args.gamma, args.gae_lambda,
                    successor_gae_fn,
                ).flatten(0, 1)
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
            updates = ppo_updates = ssl_updates = reward_updates = successor_updates = 0
            for accumulator in (*ppo_sums.values(), *world_sums.values()):
                accumulator.zero_()
            balance_metrics = {}
            measure_balance = (args.balance_interval > 0
                               and (iteration == 1 or iteration % args.balance_interval == 0))
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    batches = iter_update_batches(args.batch_size, args.minibatch_size,
                                                  args.ssl_minibatch_size, device, shuffle_generator)
                    for ppo_indices, ssl_indices in batches:
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        policy_step = ppo_indices is not None
                        for owner in optimizers:
                            if owner is not None:
                                owner.zero_grad(set_to_none=True)
                        if policy_step:
                            evaluate = diagnostic_model if measure_balance and updates == 0 else joint_model
                            ppo, components, ppo_metrics, world_metrics = evaluate(
                                b_obs[ppo_indices], b_native[ppo_indices], b_logprobs[ppo_indices],
                                b_advantages[ppo_indices], b_returns[ppo_indices], b_values[ppo_indices],
                                b_next_obs[ssl_indices], b_rewards[ssl_indices],
                                b_successor_targets[ssl_indices], b_critic_inputs[ppo_indices],
                            )
                            if measure_balance and updates == 0:
                                balance_metrics = gradient_balance(agent, components)
                            objective = ppo + components.sum()
                        else:
                            components, world_metrics = world_model(
                                b_obs[ssl_indices], b_native[ssl_indices], b_next_obs[ssl_indices],
                                b_rewards[ssl_indices], b_successor_targets[ssl_indices],
                            )
                            objective = components.sum()
                        # ALL relevant losses backward before ANY optimizer steps.
                        objective.backward()
                        optimizer_step(agent, optimizers, args, policy_step=policy_step,
                                       parameters=parameters, norms=gradient_norms[updates],
                                       projection=project_weights)
                        if policy_step:
                            for name, metric in ppo_metrics.items():
                                ppo_sums[name].add_(metric.detach())
                            ppo_updates += 1
                        for name, metric in world_metrics.items():
                            world_sums[name].add_(metric.detach())
                        ssl_updates += 1
                        reward_updates += reward_optimizer is not None
                        successor_updates += 1
                        updates += 1
                # One refresh after every learner phase includes all PPO/world
                # updates and projections, before any subsequent rollout action.
                host_actor.refresh()

            total_ppo_updates += ppo_updates
            total_ssl_updates += ssl_updates
            total_reward_updates += reward_updates
            total_successor_updates += successor_updates
            metric_values = {name: total / ppo_updates for name, total in ppo_sums.items()}
            metric_values["losses/explained_variance"] = explained_variance(b_values, b_returns)
            metric_values["gradients/ppo_norm"] = gradient_norms[:updates, 0].sum() / ppo_updates
            # Explicit owner denominators; SF exposure equals SSL exposure, not PPO.
            for name, total in world_sums.items():
                denominator = (successor_updates if name.startswith("successor/") else
                               reward_updates if name.startswith("reward/") else ssl_updates)
                metric_values[name] = total / denominator
            metric_values["gradients/ssl_norm"] = gradient_norms[:updates, 1].sum() / ssl_updates
            metric_values["gradients/successor_head_norm"] = gradient_norms[:updates, 3].sum() / successor_updates
            with timer.span("diagnostics"):
                metric_values.update(fit_diagnostics_model(
                    b_obs, b_critic_inputs, b_successors, b_successor_targets,
                ))
            if reward_updates:
                metric_values["gradients/reward_head_norm"] = gradient_norms[:updates, 2].sum() / reward_updates
            metric_values.update(balance_metrics)
            logged = gather_metrics(metric_values)
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name not in {"losses/explained_variance", "reward/explained_variance"}):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            exposure = {
                "ppo_steps": ppo_updates, "ssl_steps": ssl_updates, "reward_steps": reward_updates,
                "successor_steps": successor_updates,
                "successor_examples": successor_updates * args.ssl_minibatch_size,
                "successor_steps_total": total_successor_updates,
                "successor_examples_total": total_successor_updates * args.ssl_minibatch_size,
                "ppo_examples": ppo_updates * args.minibatch_size,
                "ssl_examples": ssl_updates * args.ssl_minibatch_size,
                "reward_examples": reward_updates * args.ssl_minibatch_size,
                "ppo_steps_total": total_ppo_updates, "ssl_steps_total": total_ssl_updates,
                "reward_steps_total": total_reward_updates,
                "ppo_examples_total": total_ppo_updates * args.minibatch_size,
                "ssl_examples_total": total_ssl_updates * args.ssl_minibatch_size,
                "reward_examples_total": total_reward_updates * args.ssl_minibatch_size,
                "ppo_minibatch_size": args.minibatch_size,
                "ssl_minibatch_size": args.ssl_minibatch_size,
                "epochs": args.update_epochs,
            }
            for name, value in exposure.items():
                writer.add_scalar(f"updates/{name}", value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            if ssl_optimizer is not None:
                writer.add_scalar("charts/ssl_learning_rate", ssl_optimizer.param_groups[0]["lr"], global_step)
            if reward_optimizer is not None:
                writer.add_scalar("charts/reward_learning_rate", reward_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/successor_learning_rate", successor_optimizer.param_groups[0]["lr"], global_step)
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
                    "global_step": global_step,
                    "parameter_counts": counts,
                    "optimizer_steps": {
                        "ppo": total_ppo_updates, "representation": total_ssl_updates,
                        "reward": total_reward_updates, "successor": total_successor_updates,
                    },
                    "exposure": exposure,
                    "optimizers": {
                        "ppo": optimizer.state_dict(),
                        "representation": None if ssl_optimizer is None else ssl_optimizer.state_dict(),
                        "reward": None if reward_optimizer is None else reward_optimizer.state_dict(),
                        "successor": successor_optimizer.state_dict(),
                    },
                    "reward_norm": {
                        "returns": torch.from_numpy(rew_norm.returns.copy()),
                        "means": torch.from_numpy(rew_norm.means.copy()),
                        "variances": torch.from_numpy(rew_norm.variances.copy()),
                        "counts": torch.from_numpy(rew_norm.counts.copy()),
                        "gamma": rew_norm.gamma,
                        "epsilon": rew_norm.epsilon,
                        "clip": rew_norm.clip,
                    },
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
