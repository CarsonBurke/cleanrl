# Conventional end-to-end-backprop streaming TD(lambda) control v1.
#
# This is the temporal-credit control for the streaming predictive-coding family:
# the Beta actor and value critic use the same six 64-wide stages, but gradients are
# propagated conventionally through the complete hierarchy. Every environment owns
# exact full-parameter actor score and critic value eligibility traces. The current
# transition's Jacobians enter the traces before its TD error is applied; traces are
# reset only after terminal/truncated transitions have contributed their final update.
# There are no rollouts, replay, PPO ratios, GAE, minibatches, or update epochs.
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions import Beta, kl_divergence
from torch.func import functional_call, grad, vmap
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
    """CUDA is required for this research variant"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = False
    """whether to save the model into the run folder"""

    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total environment transitions"""
    num_envs: int = 16
    """parallel environments and per-update sample count"""
    learning_rate: float = 3e-4
    """actor and critic base step size"""
    anneal_lr: bool = True
    """linearly anneal both base step sizes"""
    gamma: float = 0.99
    """discount factor"""
    trace_lambda: float = 0.95
    """lambda for both accumulating eligibility traces"""
    actor_trace_lambda: Optional[float] = None
    """actor lambda override; defaults to trace_lambda"""
    critic_trace_lambda: Optional[float] = None
    """critic lambda override; defaults to trace_lambda"""
    ent_coef: float = 0.0
    """direct entropy-gradient coefficient (entropy is not placed in the score trace)"""
    vf_coef: float = 0.5
    """critic update multiplier"""
    max_grad_norm: float = 0.5
    """maximum norm of each mean trace-modulated update direction"""
    td_rms_decay: float = 0.999
    """EMA decay for TD-error second moment used to scale actor updates"""
    td_norm_clip: float = 10.0
    """absolute clip after TD RMS scaling; <=0 disables"""
    td_rms_min: float = 0.1
    """minimum actor TD RMS denominator"""
    critic_td_clip: float = 10.0
    """absolute clip on raw critic TD errors; <=0 disables"""
    hidden_size: int = 64
    """hidden width matched to the predictive-coding hierarchy"""
    num_hidden_layers: int = 6
    """number of hidden stages matched to the predictive-coding hierarchy"""
    hidden_state_clip: float = 5.0
    """absolute hidden-state clamp matched to PC deployment inference; <=0 disables"""

    target_update_kl: float = 0.003
    """soft target KL on states used by each streaming actor update"""
    max_update_kl: float = 0.01
    """hard current-state KL cap enforced by actor-step bisection"""
    kl_bisection_steps: int = 10
    """bisection iterations when a proposed actor update violates max_update_kl"""
    kl_adaptation_rate: float = 0.05
    """relative-error log-space feedback rate for actor step-size control"""
    kl_scale_min: float = 0.05
    """minimum actor step multiplier"""
    kl_scale_max: float = 2.0
    """maximum actor step multiplier"""

    compile: bool = True
    """compile the batched per-sample Jacobian transforms"""
    compile_mode: Optional[str] = "reduce-overhead"
    """torch.compile mode"""
    torch_float32_matmul_precision: str = "high"
    """float32 matmul precision policy"""
    log_interval: int = 100
    """vector updates between dense diagnostics"""

    num_updates: int = 0
    """computed vector-step update count"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def beta_log_prob(alpha, beta, z):
    """Elementwise Beta log density without Distribution validation/vmap control flow."""
    z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    log_normalizer = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    return log_normalizer + (alpha - 1.0) * z.log() + (beta - 1.0) * torch.log1p(-z)


def beta_entropy(alpha, beta):
    log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return (
        log_beta
        - (alpha - 1.0) * torch.digamma(alpha)
        - (beta - 1.0) * torch.digamma(beta)
        + (alpha + beta - 2.0) * torch.digamma(alpha + beta)
    )


class DeepFeatures(nn.Module):
    """Six-stage feature stack matching BPC's feed-forward initialization map."""

    def __init__(self, input_dim, hidden_size, num_hidden_layers, hidden_state_clip):
        super().__init__()
        assert num_hidden_layers >= 1
        self.hidden_state_clip = hidden_state_clip
        self.layers = nn.ModuleList()
        for layer_idx in range(num_hidden_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_size
            self.layers.append(layer_init(nn.Linear(in_dim, hidden_size)))

    def forward(self, x, return_states=False):
        states = []
        for layer_idx, layer in enumerate(self.layers):
            # BPC's first edge consumes raw observations; every later edge consumes
            # tanh of the preceding explicit state. Its final state is unactivated.
            x = layer(x if layer_idx == 0 else torch.tanh(x))
            if self.hidden_state_clip > 0:
                x = x.clamp(-self.hidden_state_clip, self.hidden_state_clip)
            states.append(x)
        return (x, states) if return_states else x


class BetaActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size, num_hidden_layers, hidden_state_clip):
        super().__init__()
        self.features = DeepFeatures(input_dim, hidden_size, num_hidden_layers, hidden_state_clip)
        self.alpha_head = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

    def forward(self, x, return_states=False):
        if return_states:
            h, states = self.features(x, return_states=True)
        else:
            h = self.features(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        return (alpha, beta, states) if return_states else (alpha, beta)


class ValueCritic(nn.Module):
    def __init__(self, input_dim, hidden_size, num_hidden_layers, hidden_state_clip):
        super().__init__()
        self.features = DeepFeatures(input_dim, hidden_size, num_hidden_layers, hidden_state_clip)
        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x, return_states=False):
        if return_states:
            h, states = self.features(x, return_states=True)
            return self.value_head(h), states
        return self.value_head(self.features(x))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = BetaActor(
            obs_dim,
            action_dim,
            args.hidden_size,
            args.num_hidden_layers,
            args.hidden_state_clip,
        )
        self.critic = ValueCritic(
            obs_dim,
            args.hidden_size,
            args.num_hidden_layers,
            args.hidden_state_clip,
        )
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def action_from_z(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    @torch.no_grad()
    def get_action_and_value(self, x, action_z=None, return_states=False):
        if return_states:
            alpha, beta, actor_states = self.actor(x, return_states=True)
            value, critic_states = self.critic(x, return_states=True)
        else:
            alpha, beta = self.actor(x)
            value = self.critic(x)
        dist = Beta(alpha, beta)
        if action_z is None:
            action_z = dist.sample()
        action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        result = (
            self.action_from_z(action_z),
            action_z,
            beta_log_prob(alpha, beta, action_z).sum(1),
            beta_entropy(alpha, beta).sum(1),
            value,
            alpha,
            beta,
        )
        return result + (actor_states, critic_states) if return_states else result


class FlatParameterLayout:
    """Stable named-parameter flattening and telemetry grouping."""

    def __init__(self, module):
        self.names = []
        self.parameters = []
        self.slices = {}
        self.groups = OrderedDict()
        offset = 0
        for name, parameter in module.named_parameters():
            self.names.append(name)
            self.parameters.append(parameter)
            parameter_slice = slice(offset, offset + parameter.numel())
            self.slices[name] = parameter_slice
            group = self._group_name(name)
            self.groups.setdefault(group, []).append(parameter_slice)
            offset += parameter.numel()
        self.numel = offset

    @staticmethod
    def _group_name(name):
        if name.startswith("features.layers."):
            return f"layer_{int(name.split('.')[2]) + 1}"
        return "head"

    def flatten_batched(self, gradients):
        batch_size = gradients[self.names[0]].shape[0]
        return torch.cat([gradients[name].reshape(batch_size, -1) for name in self.names], dim=1)

    def flat_parameters(self):
        return torch.cat([parameter.detach().reshape(-1) for parameter in self.parameters])

    @torch.no_grad()
    def add_flat_(self, direction, step_size):
        assert direction.shape == (self.numel,)
        for name, parameter in zip(self.names, self.parameters):
            parameter.add_(direction[self.slices[name]].view_as(parameter), alpha=step_size)

    @torch.no_grad()
    def copy_flat_(self, flat):
        assert flat.shape == (self.numel,)
        for name, parameter in zip(self.names, self.parameters):
            parameter.copy_(flat[self.slices[name]].view_as(parameter))

    def group_rms(self, batched_flat):
        result = {}
        for group, slices in self.groups.items():
            values = torch.cat([batched_flat[:, parameter_slice] for parameter_slice in slices], dim=1)
            result[group] = values.square().mean().sqrt()
        return result

    def group_norm(self, flat):
        result = {}
        for group, slices in self.groups.items():
            values = torch.cat([flat[parameter_slice] for parameter_slice in slices])
            result[group] = values.norm()
        return result


class ExactPerEnvironmentJacobians:
    """Reverse-mode per-sample gradients, vectorized without cross-sample reduction."""

    def __init__(self, actor, critic, args):
        self.actor = actor
        self.critic = critic

        def actor_logprob(parameters, observation, action_z):
            alpha, beta = functional_call(actor, parameters, (observation.unsqueeze(0),))
            return beta_log_prob(alpha.squeeze(0), beta.squeeze(0), action_z).sum()

        def actor_entropy_value(parameters, observation):
            alpha, beta = functional_call(actor, parameters, (observation.unsqueeze(0),))
            return beta_entropy(alpha.squeeze(0), beta.squeeze(0)).sum()

        def critic_value(parameters, observation):
            return functional_call(critic, parameters, (observation.unsqueeze(0),)).squeeze()

        self._actor_score = vmap(grad(actor_logprob), in_dims=(None, 0, 0), randomness="error")
        self._actor_entropy = vmap(grad(actor_entropy_value), in_dims=(None, 0), randomness="error")
        self._critic_value = vmap(grad(critic_value), in_dims=(None, 0), randomness="error")
        if args.compile:
            self._actor_score = torch.compile(self._actor_score, mode=args.compile_mode, dynamic=False)
            if args.ent_coef != 0.0:
                self._actor_entropy = torch.compile(self._actor_entropy, mode=args.compile_mode, dynamic=False)
            self._critic_value = torch.compile(self._critic_value, mode=args.compile_mode, dynamic=False)

    @staticmethod
    def _parameters(module):
        return dict(module.named_parameters())

    def actor_score(self, observation, action_z):
        return self._actor_score(self._parameters(self.actor), observation, action_z)

    def actor_entropy(self, observation):
        return self._actor_entropy(self._parameters(self.actor), observation)

    def critic_value(self, observation):
        return self._critic_value(self._parameters(self.critic), observation)


class EligibilityTrace:
    """Independent accumulating full-parameter traces for a vector environment."""

    def __init__(self, num_envs, num_parameters, device):
        self.value = torch.zeros((num_envs, num_parameters), device=device)

    @torch.no_grad()
    def accumulate(self, instantaneous, decay):
        if instantaneous.shape != self.value.shape:
            raise ValueError(f"gradient shape {instantaneous.shape} does not match trace {self.value.shape}")
        self.value.mul_(decay).add_(instantaneous)
        return self.value

    def modulated_mean(self, signal):
        if signal.shape != (self.value.shape[0],):
            raise ValueError("one scalar modulator is required per environment")
        return (signal.detach().unsqueeze(1) * self.value).mean(dim=0)

    @torch.no_grad()
    def reset(self, done):
        # Avoid a host synchronization on the common no-done path.
        self.value.masked_fill_(done.unsqueeze(1), 0.0)


def clip_update_direction(direction, max_norm):
    norm = direction.norm()
    if max_norm > 0:
        direction = direction * (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    return direction, norm


class RunningTDRMS:
    """Bias-free startup followed by an EMA of E[delta^2]."""

    def __init__(self, device, decay, minimum):
        self.decay = decay
        self.minimum = minimum
        self.mean_square = torch.ones((), device=device)
        self.initialized = False

    @torch.no_grad()
    def normalize(self, delta, clip):
        batch_square = delta.square().mean()
        if self.initialized:
            self.mean_square.lerp_(batch_square, 1.0 - self.decay)
        else:
            self.mean_square.copy_(batch_square)
            self.initialized = True
        normalized = delta / self.mean_square.sqrt().clamp_min(self.minimum)
        return normalized.clamp(-clip, clip) if clip > 0 else normalized


class OnlineKLController:
    """Log-space feedback on the next actor step; does not rewrite past updates."""

    def __init__(self, target, adaptation_rate, scale_min, scale_max, device):
        if target <= 0:
            raise ValueError("target_update_kl must be positive")
        if not 0 < scale_min <= scale_max:
            raise ValueError("KL scale bounds must satisfy 0 < min <= max")
        if adaptation_rate < 0:
            raise ValueError("kl_adaptation_rate must be nonnegative")
        self.target = target
        self.adaptation_rate = adaptation_rate
        self.log_min = float(np.log(scale_min))
        self.log_max = float(np.log(scale_max))
        self.log_scale = torch.zeros((), device=device)

    @property
    def scale(self):
        return self.log_scale.exp()

    @torch.no_grad()
    def observe(self, kl):
        # Match the PC arm's bounded relative-error controller exactly. A hard
        # bisection guard separately prevents any individual catastrophic step.
        feedback = ((self.target - kl) / self.target).clamp(-2.0, 2.0)
        self.log_scale.add_(self.adaptation_rate * feedback).clamp_(self.log_min, self.log_max)


@torch.no_grad()
def apply_actor_update_with_kl_limit(
    actor,
    layout,
    observation,
    old_alpha,
    old_beta,
    direction,
    base_step_size,
    proposed_scale,
    max_kl,
    bisection_steps,
):
    """Apply an ascent step, shrinking it when its current-state KL is too large."""
    if max_kl <= 0:
        raise ValueError("max_update_kl must be positive")
    if bisection_steps < 1:
        raise ValueError("kl_bisection_steps must be positive")

    original = layout.flat_parameters()

    def apply_scale(scale):
        layout.copy_flat_(original)
        layout.add_flat_(direction, base_step_size * scale)

    def current_kl():
        new_alpha, new_beta = actor(observation)
        return kl_divergence(Beta(old_alpha, old_beta), Beta(new_alpha, new_beta)).sum(1).mean()

    proposal = float(proposed_scale.detach().item() if torch.is_tensor(proposed_scale) else proposed_scale)
    apply_scale(proposal)
    accepted_kl = current_kl()
    if accepted_kl.item() <= max_kl:
        return accepted_kl, proposal, False

    low, high = 0.0, proposal
    accepted_kl = torch.zeros_like(accepted_kl)
    for _ in range(bisection_steps):
        candidate = 0.5 * (low + high)
        apply_scale(candidate)
        candidate_kl = current_kl()
        if candidate_kl.item() <= max_kl:
            low = candidate
            accepted_kl = candidate_kl
        else:
            high = candidate

    # The last candidate may have been rejected, so explicitly commit the largest
    # accepted endpoint. KL at scale zero is exactly zero if no positive step fit.
    apply_scale(low)
    return accepted_kl, low, True


def bootstrap_observations(next_obs, truncations, infos):
    """Replace autoreset observations by the time-limit final observations."""
    bootstrap_obs = np.array(next_obs, copy=True)
    if not np.any(truncations):
        return bootstrap_obs
    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")
    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        if final_observations[env_idx] is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observations[env_idx]
    return bootstrap_obs


def main():
    args = tyro.cli(Args)
    args.num_updates = args.total_timesteps // args.num_envs
    actor_lambda = args.trace_lambda if args.actor_trace_lambda is None else args.actor_trace_lambda
    critic_lambda = args.trace_lambda if args.critic_trace_lambda is None else args.critic_trace_lambda
    if not 0.0 <= actor_lambda <= 1.0 or not 0.0 <= critic_lambda <= 1.0:
        raise ValueError("trace lambdas must be in [0, 1]")

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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)
    assert args.cuda and torch.cuda.is_available(), "CUDA is required for this research variant"
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous actions are supported"
    agent = Agent(envs, args).to(device)
    actor_layout = FlatParameterLayout(agent.actor)
    critic_layout = FlatParameterLayout(agent.critic)
    jacobians = ExactPerEnvironmentJacobians(agent.actor, agent.critic, args)
    actor_trace = EligibilityTrace(args.num_envs, actor_layout.numel, device)
    critic_trace = EligibilityTrace(args.num_envs, critic_layout.numel, device)
    td_rms = RunningTDRMS(device, args.td_rms_decay, args.td_rms_min)
    kl_controller = OnlineKLController(
        args.target_update_kl,
        args.kl_adaptation_rate,
        args.kl_scale_min,
        args.kl_scale_max,
        device,
    )

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    for update in range(1, args.num_updates + 1):
        global_step += args.num_envs
        frac = 1.0 - (update - 1.0) / args.num_updates if args.anneal_lr else 1.0
        actor_base_lr = args.learning_rate * frac
        critic_lr = args.learning_rate * frac
        obs = next_obs

        with torch.no_grad():
            (
                action,
                action_z,
                logprob,
                entropy,
                value,
                old_alpha,
                old_beta,
                actor_states,
                critic_states,
            ) = agent.get_action_and_value(obs, return_states=True)
            value = value.view(-1)

        next_obs_np, reward_np, terminations_np, truncations_np, infos = envs.step(action.cpu().numpy())
        bootstrap_obs_np = bootstrap_observations(next_obs_np, truncations_np, infos)
        bootstrap_obs = torch.as_tensor(bootstrap_obs_np, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(terminations_np, dtype=torch.bool, device=device)
        truncated = torch.as_tensor(truncations_np, dtype=torch.bool, device=device)
        done = terminated | truncated

        # Both predictions and both exact Jacobians use one pre-update parameter state.
        # Only true MDP termination suppresses bootstrapping; truncation uses final_obs.
        with torch.no_grad():
            next_value = agent.critic(bootstrap_obs).view(-1)
            td_target = reward + args.gamma * (~terminated).float() * next_value
            td_error = td_target - value
            actor_td = td_rms.normalize(td_error, args.td_norm_clip)
            critic_td = td_error.clamp(-args.critic_td_clip, args.critic_td_clip) if args.critic_td_clip > 0 else td_error

        # torch.func gradients are higher-order differentiable by default. Eligibility
        # is state, not a meta-gradient graph, so detach immediately to avoid retaining
        # an unnecessary second-order graph across parameter mutation.
        actor_instantaneous = actor_layout.flatten_batched(jacobians.actor_score(obs, action_z)).detach()
        critic_instantaneous = critic_layout.flatten_batched(jacobians.critic_value(obs)).detach()
        actor_trace.accumulate(actor_instantaneous, args.gamma * actor_lambda)
        critic_trace.accumulate(critic_instantaneous, args.gamma * critic_lambda)

        # The current eligibility is present before delta modulation. Positive delta
        # ascends log pi and V; the critic is a conventional semi-gradient TD(lambda).
        actor_direction = actor_trace.modulated_mean(actor_td)
        if args.ent_coef != 0.0:
            entropy_gradient = actor_layout.flatten_batched(jacobians.actor_entropy(obs)).mean(dim=0).detach()
            actor_direction = actor_direction + args.ent_coef * entropy_gradient
        critic_direction = args.vf_coef * critic_trace.modulated_mean(critic_td)
        actor_direction, actor_unclipped_norm = clip_update_direction(actor_direction, args.max_grad_norm)
        critic_direction, critic_unclipped_norm = clip_update_direction(critic_direction, args.max_grad_norm)

        post_update_kl, accepted_kl_scale, kl_was_limited = apply_actor_update_with_kl_limit(
            agent.actor,
            actor_layout,
            obs,
            old_alpha,
            old_beta,
            actor_direction,
            actor_base_lr,
            kl_controller.scale,
            args.max_update_kl,
            args.kl_bisection_steps,
        )
        critic_layout.add_flat_(critic_direction, critic_lr)

        kl_controller.observe(post_update_kl)

        # Terminal TD is applied first. Both termination and truncation then sever
        # temporal credit before the vector slot's autoreset episode begins.
        actor_trace.reset(done)
        critic_trace.reset(done)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = float(np.asarray(info["episode"]["r"]))
                    episodic_length = int(np.asarray(info["episode"]["l"]))
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        if update % args.log_interval == 0 or update == 1:
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/base_learning_rate", args.learning_rate * frac, global_step)
            writer.add_scalar("charts/actor_learning_rate", actor_base_lr * accepted_kl_scale, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/td_error_mean", td_error.mean().item(), global_step)
            writer.add_scalar("losses/td_error_rms", td_rms.mean_square.sqrt().item(), global_step)
            writer.add_scalar("losses/policy_surrogate", -(actor_td * logprob).mean().item(), global_step)
            writer.add_scalar("losses/value_semi_gradient", 0.5 * td_error.square().mean().item(), global_step)
            writer.add_scalar("policy/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("policy/alpha_mean", old_alpha.mean().item(), global_step)
            writer.add_scalar("policy/beta_mean", old_beta.mean().item(), global_step)
            writer.add_scalar("policy/post_update_kl", post_update_kl.item(), global_step)
            writer.add_scalar("policy/kl_step_scale", kl_controller.scale.item(), global_step)
            writer.add_scalar("policy/accepted_kl_step_scale", accepted_kl_scale, global_step)
            writer.add_scalar("policy/kl_limited", float(kl_was_limited), global_step)
            writer.add_scalar("trace/actor_rms", actor_trace.value.square().mean().sqrt().item(), global_step)
            writer.add_scalar("trace/critic_rms", critic_trace.value.square().mean().sqrt().item(), global_step)
            writer.add_scalar("trace/actor_max_abs", actor_trace.value.abs().max().item(), global_step)
            writer.add_scalar("trace/critic_max_abs", critic_trace.value.abs().max().item(), global_step)
            writer.add_scalar("trace/reset_fraction", done.float().mean().item(), global_step)
            writer.add_scalar("update/actor_unclipped_norm", actor_unclipped_norm.item(), global_step)
            writer.add_scalar("update/critic_unclipped_norm", critic_unclipped_norm.item(), global_step)
            actor_trace_groups = actor_layout.group_rms(actor_trace.value)
            critic_trace_groups = critic_layout.group_rms(critic_trace.value)
            actor_instant_groups = actor_layout.group_rms(actor_instantaneous)
            critic_instant_groups = critic_layout.group_rms(critic_instantaneous)
            actor_update_groups = actor_layout.group_norm(actor_direction)
            critic_update_groups = critic_layout.group_norm(critic_direction)
            for group in actor_trace_groups:
                writer.add_scalar(f"trace_layers/actor_{group}_rms", actor_trace_groups[group].item(), global_step)
                writer.add_scalar(f"jacobian_layers/actor_{group}_rms", actor_instant_groups[group].item(), global_step)
                writer.add_scalar(f"update_layers/actor_{group}_norm", actor_update_groups[group].item(), global_step)
            for group in critic_trace_groups:
                writer.add_scalar(f"trace_layers/critic_{group}_rms", critic_trace_groups[group].item(), global_step)
                writer.add_scalar(f"jacobian_layers/critic_{group}_rms", critic_instant_groups[group].item(), global_step)
                writer.add_scalar(f"update_layers/critic_{group}_norm", critic_update_groups[group].item(), global_step)
            for layer_idx, (actor_state, critic_state) in enumerate(zip(actor_states, critic_states), start=1):
                writer.add_scalar(f"activations/actor_layer_{layer_idx}_rms", actor_state.square().mean().sqrt().item(), global_step)
                writer.add_scalar(f"activations/critic_layer_{layer_idx}_rms", critic_state.square().mean().sqrt().item(), global_step)
            print(
                f"update={update}, global_step={global_step}, SPS={sps}, "
                f"td_rms={td_rms.mean_square.sqrt().item():.3f}, kl={post_update_kl.item():.2e}, "
                f"kl_scale={kl_controller.scale.item():.3f}"
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
