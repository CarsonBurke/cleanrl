# Streaming Energy-Based Prediction actor-critic, TD(0) prototype v1.
#
# One vector transition is followed immediately by one detached one-step TD update;
# there are no rollouts, GAE, PPO ratios, clipping, minibatches, or update epochs.
# Six explicit hidden states settle under adjacent Gaussian prediction energies and
# a terminal policy/value energy. Detached settled states train only adjacent local
# predictors, so task error is never backpropagated end-to-end through the hierarchy.
# State-inference energies are summed across examples (and averaged only for parameter
# learning), making each example's settling dynamics invariant to num_envs.
#
# This intentionally implements honest TD(0), not an approximate TD(lambda) trace:
# exact nonlinear per-environment parameter traces require B x P Jacobian state. A
# later v1 can add that machinery after the streaming/local-learning premise works.
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import kl_divergence
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
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
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
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the local/head parameter gradient clipping"""
    td_rms_decay: float = 0.999
    """EMA decay for the detached TD-error second moment used to scale actor updates"""
    td_norm_clip: float = 10.0
    """absolute clip applied after TD RMS scaling; <=0 disables clipping"""
    td_rms_min: float = 0.1
    """minimum TD RMS denominator"""
    log_interval: int = 100
    """vector updates between dense diagnostic writes"""

    hidden_size: int = 64
    """predictive-coding hidden state width"""
    pc_num_hidden_layers: int = 6
    """number of predictive-coding hidden states"""
    pc_inference_steps: int = 6
    """activation settling steps during each streaming update"""
    pc_rollout_inference_steps: int = 0
    """local-energy-only settling steps during rollout action selection"""
    pc_inference_lr: float = 0.01
    """gradient descent step size for activation-state inference"""
    pc_activation_grad_clip: float = 1.0
    """per-state activation gradient norm clip; <=0 disables clipping"""
    pc_state_clip: float = 3.0
    """default absolute clamp for inferred hidden states; per-layer overrides below can refine it"""
    pc_state_clip_h1: Optional[float] = None
    """absolute clamp for the first inferred hidden state; defaults to pc_state_clip"""
    pc_state_clip_h2: Optional[float] = None
    """absolute clamp for hidden states after h1; defaults to pc_state_clip"""
    pc_local_energy_coef: float = 1.0
    """coefficient on local predictive energies during settling and local learning"""
    pc_actor_terminal_coef: float = 1.0
    """coefficient on terminal policy energy during actor-state settling"""
    pc_critic_terminal_coef: float = 1.0
    """coefficient on terminal value energy during critic-state settling"""
    pc_fixed_logvar: float = 0.0
    """initial/fixed local predictor log variance"""
    pc_learn_logvar: bool = False
    """learn local predictor log variances; false uses fixed log variance"""
    pc_logvar_min: float = -2.0
    """minimum local predictor log variance"""
    pc_logvar_max: float = 2.0
    """maximum local predictor log variance"""
    pc_input_activation: str = "identity"
    """activation f(x) used by the first BPC local predictor: identity, tanh, silu, relusq"""
    pc_hidden_activation: str = "tanh"
    """activation f(h) used by hidden-to-hidden BPC local predictors: identity, tanh, silu, relusq"""

    compile: bool = True
    """compile pure modules with torch.compile"""
    compile_mode: Optional[str] = "reduce-overhead"
    """torch.compile mode"""
    compile_disable_cudagraphs: bool = True
    """disable Inductor CUDA graphs; activation autograd.grad reuses compiled outputs too aggressively"""
    torch_float32_matmul_precision: str = "high"
    """float32 matmul precision policy"""

    # to be filled in runtime
    num_updates: int = 0
    """the number of vector-step updates (computed in runtime)"""


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
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def clip_state_grad(grad, max_norm):
    if max_norm <= 0:
        return grad
    norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (max_norm / norm).clamp(max=1.0)
    return grad * scale


def predictor_activation(x, name):
    if name == "identity":
        return x
    if name == "tanh":
        return torch.tanh(x)
    if name == "silu":
        return F.silu(x)
    if name == "relusq":
        return F.relu(x).square()
    raise ValueError(f"unknown predictor activation: {name}")


def clip_state_value(state, max_abs):
    if max_abs <= 0:
        return state
    return state.clamp(-max_abs, max_abs)


def state_clip_for_layer(args, layer_idx):
    override = args.pc_state_clip_h1 if layer_idx == 0 else args.pc_state_clip_h2
    return args.pc_state_clip if override is None else override


def clone_graph_output(output):
    if torch.is_tensor(output):
        return output.clone()
    if isinstance(output, tuple):
        return tuple(clone_graph_output(item) for item in output)
    if isinstance(output, list):
        return [clone_graph_output(item) for item in output]
    if isinstance(output, dict):
        return {key: clone_graph_output(value) for key, value in output.items()}
    return output


class CompiledModule(nn.Module):
    def __init__(self, module, args):
        super().__init__()
        self._orig_mod = module
        self.disable_cudagraphs = args.compile_disable_cudagraphs
        if self.disable_cudagraphs:
            import torch._inductor as inductor

            # mode="reduce-overhead" normally re-enables CUDA graphs via a
            # per-compile patch, overriding the global setting. Preserve the
            # rest of the mode while explicitly disabling that unsafe option.
            compile_options = dict(inductor.list_mode_options(args.compile_mode, dynamic=False))
            compile_options["triton.cudagraphs"] = False
            self._compiled_forward = torch.compile(
                module.forward,
                dynamic=False,
                options=compile_options,
            )
        else:
            self._compiled_forward = torch.compile(
                module.forward,
                mode=args.compile_mode,
                dynamic=False,
            )

    def forward(self, *args, **kwargs):
        if not self.disable_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        output = self._compiled_forward(*args, **kwargs)
        if not self.disable_cudagraphs:
            output = clone_graph_output(output)
        return output


def maybe_compile(module, args):
    if not args.compile:
        return module
    if not hasattr(torch, "compile"):
        print(f"torch.compile unavailable; running {module.__class__.__name__} eagerly")
        return module
    return CompiledModule(module, args)


def clean_compiled_state_dict(module):
    state_dict = module.state_dict()
    return {key.replace("._orig_mod", ""): value for key, value in state_dict.items()}


class LocalPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, fixed_logvar=0.0, learn_logvar=False, activation="identity"):
        super().__init__()
        self.activation = activation
        self.mean = layer_init(nn.Linear(in_dim, out_dim))
        self.learn_logvar = learn_logvar
        if learn_logvar:
            self.logvar = layer_init(nn.Linear(in_dim, out_dim), std=0.01, bias_const=fixed_logvar)
        else:
            self.register_buffer("fixed_logvar", torch.full((out_dim,), fixed_logvar))

    def forward(self, x, logvar_min, logvar_max):
        features = predictor_activation(x, self.activation)
        mean = self.mean(features)
        if self.learn_logvar:
            logvar = self.logvar(features).clamp(logvar_min, logvar_max)
        else:
            logvar = self.fixed_logvar.expand_as(mean).clamp(logvar_min, logvar_max)
        return mean, logvar


class PCHierarchy(nn.Module):
    def __init__(self, input_dim, hidden_size, args):
        super().__init__()
        assert args.pc_num_hidden_layers >= 1, "pc_num_hidden_layers must be >= 1"
        edges = [
            LocalPredictor(
                input_dim,
                hidden_size,
                args.pc_fixed_logvar,
                args.pc_learn_logvar,
                args.pc_input_activation,
            )
        ]
        for _ in range(1, args.pc_num_hidden_layers):
            edges.append(
                LocalPredictor(
                    hidden_size,
                    hidden_size,
                    args.pc_fixed_logvar,
                    args.pc_learn_logvar,
                    args.pc_hidden_activation,
                )
            )
        self.edges = nn.ModuleList(edges)

    def initial_states(self, x, args):
        states = []
        source = x
        for layer_idx, edge in enumerate(self.edges):
            mean, _ = edge(source, args.pc_logvar_min, args.pc_logvar_max)
            mean = clip_state_value(mean, state_clip_for_layer(args, layer_idx))
            states.append(mean)
            source = mean
        return states

    def local_energies_per_sample(self, x, states, args):
        assert len(states) == len(self.edges), "state count must match predictive edge count"
        energies = []
        source = x
        for state, edge in zip(states, self.edges):
            mean, logvar = edge(source, args.pc_logvar_min, args.pc_logvar_max)
            energy = 0.5 * (((state - mean) ** 2) * torch.exp(-logvar) + logvar)
            energies.append(energy.mean(dim=1))
            source = state
        return energies

    def local_learning_loss(self, x, states, args):
        per_sample = sum(self.local_energies_per_sample(x.detach(), [state.detach() for state in states], args))
        return per_sample.mean()


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
        scale = self.mean_square.sqrt().clamp_min(self.minimum)
        normalized = delta / scale
        return normalized.clamp(-clip, clip) if clip > 0 else normalized


def bootstrap_observations(next_obs, truncations, infos):
    """Replace Gymnasium autoreset observations with time-limit final observations."""
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
        final_obs = final_observations[env_idx]
        if final_obs is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_obs
    return bootstrap_obs


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.actor_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.critic_pc = PCHierarchy(obs_dim, args.hidden_size, args)
        self.actor_alpha_head = layer_init(nn.Linear(args.hidden_size, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(args.hidden_size, act_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(args.hidden_size, 1), std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def compile_modules(self, args):
        for idx, edge in enumerate(self.actor_pc.edges):
            self.actor_pc.edges[idx] = maybe_compile(edge, args)
        for idx, edge in enumerate(self.critic_pc.edges):
            self.critic_pc.edges[idx] = maybe_compile(edge, args)
        self.actor_alpha_head = maybe_compile(self.actor_alpha_head, args)
        self.actor_beta_head = maybe_compile(self.actor_beta_head, args)
        self.critic_head = maybe_compile(self.critic_head, args)

    def local_parameters(self):
        yield from self.actor_pc.parameters()
        yield from self.critic_pc.parameters()

    def head_parameters(self):
        yield from self.actor_alpha_head.parameters()
        yield from self.actor_beta_head.parameters()
        yield from self.critic_head.parameters()

    def actor_dist_from_state(self, h):
        alpha = 1.0 + F.softplus(self.actor_alpha_head(h))
        beta = 1.0 + F.softplus(self.actor_beta_head(h))
        return Beta(alpha, beta)

    def action_from_z(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def value_from_state(self, h):
        return self.critic_head(h)

    def get_value(self, x, args):
        states = self.deployment_states(self.critic_pc, x, args)
        return self.value_from_state(states[-1])

    def get_action_and_value(self, x, args, action_z=None):
        actor_states = self.deployment_states(self.actor_pc, x, args)
        critic_states = self.deployment_states(self.critic_pc, x, args)
        dist = self.actor_dist_from_state(actor_states[-1])
        if action_z is None:
            action_z = dist.sample()
        action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        log_prob = dist.log_prob(action_z).sum(1)
        entropy = dist.entropy().sum(1)
        return self.action_from_z(action_z), action_z, log_prob, entropy, self.value_from_state(critic_states[-1])

    def deployment_states(self, hierarchy, x, args):
        with torch.no_grad():
            states = hierarchy.initial_states(x, args)
        if args.pc_rollout_inference_steps > 0:
            with torch.enable_grad():
                states, _ = self.settle_states(
                    hierarchy,
                    x.detach(),
                    [state.detach() for state in states],
                    args,
                    terminal_energy_fn=None,
                    steps=args.pc_rollout_inference_steps,
                )
        return [state.detach() for state in states]

    def actor_terminal_energy_per_sample(self, h, action_z, td_signal, args):
        dist = self.actor_dist_from_state(h)
        action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        logprob = dist.log_prob(action_z.detach()).sum(1)
        entropy = dist.entropy().sum(1)
        return -td_signal.detach() * logprob - args.ent_coef * entropy

    def critic_terminal_energy_per_sample(self, h, td_target, args):
        newvalue = self.value_from_state(h).view(-1)
        return 0.5 * (newvalue - td_target.detach()).square()

    def settle_states(self, hierarchy, x, initial_states, args, terminal_energy_fn, steps=None):
        steps = args.pc_inference_steps if steps is None else steps
        states = [state.detach().requires_grad_(True) for state in initial_states]
        first_energy = None
        last_energy = None
        for _ in range(steps):
            local_e = args.pc_local_energy_coef * sum(hierarchy.local_energies_per_sample(x.detach(), states, args))
            terminal_e = torch.zeros_like(local_e)
            if terminal_energy_fn is not None:
                terminal_e = terminal_energy_fn(states[-1])
            # Sum independent per-example energies: gradients for each activation row
            # do not shrink when num_envs changes.
            energy = (local_e + terminal_e).sum()
            if first_energy is None:
                first_energy = energy.detach()
            grads = torch.autograd.grad(energy, states, create_graph=False, retain_graph=False)
            next_states = []
            for layer_idx, (state, grad) in enumerate(zip(states, grads)):
                grad = clip_state_grad(grad, args.pc_activation_grad_clip)
                next_state = state - args.pc_inference_lr * grad
                next_state = clip_state_value(next_state, state_clip_for_layer(args, layer_idx))
                next_states.append(next_state.detach().requires_grad_(True))
            states = next_states
        if steps > 0:
            final_local_e = args.pc_local_energy_coef * sum(
                hierarchy.local_energies_per_sample(x.detach(), states, args)
            )
            final_terminal_e = torch.zeros_like(final_local_e)
            if terminal_energy_fn is not None:
                final_terminal_e = terminal_energy_fn(states[-1])
            last_energy = (final_local_e + final_terminal_e).sum().detach()
        else:
            local_e = args.pc_local_energy_coef * sum(hierarchy.local_energies_per_sample(x.detach(), states, args))
            terminal_e = torch.zeros_like(local_e)
            if terminal_energy_fn is not None:
                terminal_e = terminal_energy_fn(states[-1])
            first_energy = last_energy = (local_e + terminal_e).sum().detach()
        return [state.detach() for state in states], (first_energy, last_energy)

    def infer_actor_update_states(self, x, action_z, td_signal, args):
        with torch.no_grad():
            initial_states = self.actor_pc.initial_states(x, args)

        def terminal(h):
            return args.pc_actor_terminal_coef * self.actor_terminal_energy_per_sample(h, action_z, td_signal, args)

        return self.settle_states(self.actor_pc, x, initial_states, args, terminal)

    def infer_critic_update_states(self, x, td_target, args):
        with torch.no_grad():
            initial_states = self.critic_pc.initial_states(x, args)

        def terminal(h):
            return args.pc_critic_terminal_coef * self.critic_terminal_energy_per_sample(h, td_target, args)

        return self.settle_states(self.critic_pc, x, initial_states, args, terminal)


def main():
    args = tyro.cli(Args)
    args.num_updates = args.total_timesteps // args.num_envs
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
    if args.compile:
        try:
            import torch._dynamo as dynamo

            dynamo.config.cache_size_limit = 256
        except Exception:
            pass
        if args.compile_disable_cudagraphs:
            try:
                import torch._inductor.config as inductor_config

                inductor_config.triton.cudagraphs = False
            except Exception:
                pass

    assert args.cuda and torch.cuda.is_available(), "CUDA is required for this research variant"
    device = torch.device("cuda")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    agent.compile_modules(args)
    actor_head_parameters = list(agent.actor_alpha_head.parameters()) + list(agent.actor_beta_head.parameters())
    critic_head_parameters = list(agent.critic_head.parameters())
    actor_local_parameters = list(agent.actor_pc.parameters())
    critic_local_parameters = list(agent.critic_pc.parameters())
    actor_head_optimizer = optim.Adam(actor_head_parameters, lr=args.learning_rate, eps=1e-5)
    critic_head_optimizer = optim.Adam(critic_head_parameters, lr=args.learning_rate, eps=1e-5)
    actor_local_optimizer = optim.Adam(actor_local_parameters, lr=args.learning_rate, eps=1e-5)
    critic_local_optimizer = optim.Adam(critic_local_parameters, lr=args.learning_rate, eps=1e-5)
    optimizers = (
        actor_head_optimizer,
        critic_head_optimizer,
        actor_local_optimizer,
        critic_local_optimizer,
    )
    td_rms = RunningTDRMS(device, args.td_rms_decay, args.td_rms_min)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    for update in range(1, args.num_updates + 1):
        global_step += args.num_envs
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        obs = next_obs
        with torch.no_grad():
            action, action_z, _, _, value = agent.get_action_and_value(obs, args)
            value = value.view(-1)

        next_obs_np, reward_np, terminations_np, truncations_np, infos = envs.step(action.cpu().numpy())
        bootstrap_obs_np = bootstrap_observations(next_obs_np, truncations_np, infos)
        bootstrap_obs = torch.as_tensor(bootstrap_obs_np, dtype=torch.float32, device=device)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(terminations_np, dtype=torch.float32, device=device)

        # Both values are computed before any parameter mutation. Only true termination
        # suppresses bootstrap; a time limit uses its final observation above.
        with torch.no_grad():
            next_value = agent.get_value(bootstrap_obs, args).view(-1)
            td_target = reward + args.gamma * (1.0 - terminated) * next_value
            td_error = td_target - value
            actor_td = td_rms.normalize(td_error, args.td_norm_clip)

        # Settle both pathways and cache all deployment states before updating any
        # parameters; otherwise the actor and critic would see different transitions.
        deploy_actor_states = agent.deployment_states(agent.actor_pc, obs, args)
        deploy_critic_states = agent.deployment_states(agent.critic_pc, obs, args)
        actor_states, actor_energy = agent.infer_actor_update_states(obs, action_z, actor_td, args)
        critic_states, critic_energy = agent.infer_critic_update_states(obs, td_target, args)
        with torch.no_grad():
            old_dist = agent.actor_dist_from_state(deploy_actor_states[-1])
            old_alpha = old_dist.concentration1.clone()
            old_beta = old_dist.concentration0.clone()

        dist = agent.actor_dist_from_state(deploy_actor_states[-1].detach())
        logprob = dist.log_prob(action_z.detach().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)).sum(1)
        entropy = dist.entropy().sum(1)
        actor_head_loss = -(actor_td * logprob).mean() - args.ent_coef * entropy.mean()
        critic_prediction = agent.value_from_state(deploy_critic_states[-1].detach()).view(-1)
        critic_head_loss = 0.5 * (critic_prediction - td_target).square().mean() * args.vf_coef
        actor_local_loss = args.pc_local_energy_coef * agent.actor_pc.local_learning_loss(obs, actor_states, args)
        critic_local_loss = args.pc_local_energy_coef * agent.critic_pc.local_learning_loss(obs, critic_states, args)

        actor_head_optimizer.zero_grad()
        actor_head_loss.backward()
        actor_head_grad = nn.utils.clip_grad_norm_(actor_head_parameters, args.max_grad_norm)
        actor_head_optimizer.step()

        critic_head_optimizer.zero_grad()
        critic_head_loss.backward()
        critic_head_grad = nn.utils.clip_grad_norm_(critic_head_parameters, args.max_grad_norm)
        critic_head_optimizer.step()

        actor_local_optimizer.zero_grad()
        actor_local_loss.backward()
        actor_local_grad = nn.utils.clip_grad_norm_(actor_local_parameters, args.max_grad_norm)
        actor_local_optimizer.step()

        critic_local_optimizer.zero_grad()
        critic_local_loss.backward()
        critic_local_grad = nn.utils.clip_grad_norm_(critic_local_parameters, args.max_grad_norm)
        critic_local_optimizer.step()

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
            with torch.no_grad():
                new_actor_states = agent.deployment_states(agent.actor_pc, obs, args)
                new_dist = agent.actor_dist_from_state(new_actor_states[-1])
                reference_dist = Beta(old_alpha, old_beta)
                post_update_kl = kl_divergence(reference_dist, new_dist).sum(1).mean()
                actor_displacement = torch.stack(
                    [(settled - deployed).norm(dim=1).mean() for settled, deployed in zip(actor_states, deploy_actor_states)]
                ).mean()
                critic_displacement = torch.stack(
                    [(settled - deployed).norm(dim=1).mean() for settled, deployed in zip(critic_states, deploy_critic_states)]
                ).mean()
                actor_clip_fraction = torch.stack(
                    [
                        (settled.abs() >= state_clip_for_layer(args, idx) - 1e-6).float().mean()
                        for idx, settled in enumerate(actor_states)
                    ]
                ).mean()
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/learning_rate", actor_head_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", actor_head_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", critic_head_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/td_error_mean", td_error.mean().item(), global_step)
            writer.add_scalar("losses/td_error_rms", td_rms.mean_square.sqrt().item(), global_step)
            writer.add_scalar("losses/post_update_kl", post_update_kl.item(), global_step)
            writer.add_scalar("pc/actor_local_loss", actor_local_loss.item(), global_step)
            writer.add_scalar("pc/critic_local_loss", critic_local_loss.item(), global_step)
            writer.add_scalar("pc/actor_energy_drop_per_env", (actor_energy[0] - actor_energy[1]).item() / args.num_envs, global_step)
            writer.add_scalar("pc/critic_energy_drop_per_env", (critic_energy[0] - critic_energy[1]).item() / args.num_envs, global_step)
            writer.add_scalar("pc/actor_state_displacement", actor_displacement.item(), global_step)
            writer.add_scalar("pc/critic_state_displacement", critic_displacement.item(), global_step)
            writer.add_scalar("pc/actor_state_clip_fraction", actor_clip_fraction.item(), global_step)
            writer.add_scalar("grad/actor_head", float(actor_head_grad), global_step)
            writer.add_scalar("grad/critic_head", float(critic_head_grad), global_step)
            writer.add_scalar("grad/actor_local", float(actor_local_grad), global_step)
            writer.add_scalar("grad/critic_local", float(critic_local_grad), global_step)
            print(f"update={update}, global_step={global_step}, SPS={sps}, td_rms={td_rms.mean_square.sqrt().item():.3f}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(clean_compiled_state_dict(agent), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
