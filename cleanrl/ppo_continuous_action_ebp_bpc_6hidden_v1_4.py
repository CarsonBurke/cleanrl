# PPO + Energy-Based Prediction BPC-6Hidden v1.4.
#
# Predictive-coding variant of base continuous PPO. Hidden activations are explicit
# state variables, not an ordinary end-to-end MLP path. During PPO updates, actor
# and critic states settle by minimizing local Gaussian prediction energies plus
# terminal PPO/value energy. After settling, states are detached: local predictors
# learn only adjacent settled-state prediction, and output heads learn only the
# terminal PPO/value objectives from detached terminal states.
#
# v1.4 keeps v1.2's BPC-style dense Gaussian local predictors and increases the
# predictive-coding hidden hierarchy from two hidden states to six. Local energy
# is intentionally not downscaled: six hidden states exert six local prediction
# terms. All hidden transitions after obs->h1 use the same source feature
# activation and grouped clip policy; actor/value heads read the final hidden
# state.
#
# v1.2 keeps v1's dense local Gaussian predictors but switches their mean model
# toward the Bayesian predictive-coding form mean = W f(previous_state) + b:
# there is no tanh on the predicted mean. The first predictor uses normalized
# observations directly; the second predicts from tanh(h1). Hidden states have
# separate per-layer clips so the unbounded Gaussian means cannot destabilize the
# activation-inference loop.
#
# Hypothesis: terminal task pressure can shape hidden representations through
# predictive-coding state inference while avoiding full chain-rule parameter
# backprop through the hidden hierarchy. The actor uses v168's bounded unimodal
# Beta policy, alpha/beta = 1 + softplus(head), sampled in native z in (0, 1).
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
    num_steps: int = 2048
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
    """the maximum norm for the local/head parameter gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    hidden_size: int = 64
    """predictive-coding hidden state width"""
    pc_num_hidden_layers: int = 6
    """number of predictive-coding hidden states"""
    pc_inference_steps: int = 6
    """activation settling steps during PPO minibatch updates"""
    pc_rollout_inference_steps: int = 0
    """local-energy-only settling steps during rollout action selection"""
    pc_inference_lr: float = 0.08
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
    """coefficient on terminal PPO energy during actor-state settling"""
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

    torch_compile: bool = True
    """compile pure modules with torch.compile"""
    torch_compile_mode: Optional[str] = "reduce-overhead"
    """torch.compile mode"""
    torch_compile_disable_cudagraphs: bool = True
    """disable Inductor CUDA graphs; activation autograd.grad reuses compiled outputs too aggressively"""
    torch_float32_matmul_precision: str = "high"
    """float32 matmul precision policy"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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
        self.disable_cudagraphs = args.torch_compile_disable_cudagraphs
        if self.disable_cudagraphs:
            self._compiled_forward = torch.compile(
                module.forward,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        else:
            self._compiled_forward = torch.compile(
                module.forward,
                mode=args.torch_compile_mode,
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
    if not args.torch_compile:
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

    def local_energies(self, x, states, args):
        assert len(states) == len(self.edges), "state count must match predictive edge count"
        energies = []
        source = x
        for state, edge in zip(states, self.edges):
            mean, logvar = edge(source, args.pc_logvar_min, args.pc_logvar_max)
            energy = 0.5 * (((state - mean) ** 2) * torch.exp(-logvar) + logvar)
            energies.append(energy.mean())
            source = state
        return energies

    def local_learning_loss(self, x, states, args):
        return sum(self.local_energies(x.detach(), [state.detach() for state in states], args))


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

    def ppo_terminal_energy(self, h, action_z, old_logprob, advantages, args):
        dist = self.actor_dist_from_state(h)
        action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        newlogprob = dist.log_prob(action_z.detach()).sum(1)
        logratio = newlogprob - old_logprob.detach()
        ratio = logratio.exp()
        adv = advantages.detach()
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        entropy = dist.entropy().sum(1)
        return torch.max(pg_loss1, pg_loss2).mean() - args.ent_coef * entropy.mean()

    def value_terminal_energy(self, h, returns, old_values, args):
        newvalue = self.value_from_state(h).view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - returns.detach()) ** 2
            v_clipped = old_values.detach() + torch.clamp(
                newvalue - old_values.detach(),
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns.detach()) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - returns.detach()) ** 2).mean()
        return v_loss

    def settle_states(self, hierarchy, x, initial_states, args, terminal_energy_fn, steps=None):
        steps = args.pc_inference_steps if steps is None else steps
        states = [state.detach().requires_grad_(True) for state in initial_states]
        first_energy = None
        last_energy = None
        for _ in range(steps):
            local_e = args.pc_local_energy_coef * sum(hierarchy.local_energies(x.detach(), states, args))
            terminal_e = local_e.new_zeros(())
            if terminal_energy_fn is not None:
                terminal_e = terminal_energy_fn(states[-1])
            energy = local_e + terminal_e
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
            last_energy = energy.detach()
        if steps == 0:
            local_e = args.pc_local_energy_coef * sum(hierarchy.local_energies(x.detach(), states, args))
            terminal_e = local_e.new_zeros(())
            if terminal_energy_fn is not None:
                terminal_e = terminal_energy_fn(states[-1])
            first_energy = last_energy = (local_e + terminal_e).detach()
        return [state.detach() for state in states], (first_energy, last_energy)

    def infer_actor_update_states(self, x, action_z, old_logprob, advantages, args):
        with torch.no_grad():
            initial_states = self.actor_pc.initial_states(x, args)

        def terminal(h):
            return args.pc_actor_terminal_coef * self.ppo_terminal_energy(h, action_z, old_logprob, advantages, args)

        return self.settle_states(self.actor_pc, x, initial_states, args, terminal)

    def infer_critic_update_states(self, x, returns, old_values, args):
        with torch.no_grad():
            initial_states = self.critic_pc.initial_states(x, args)

        def terminal(h):
            return args.pc_critic_terminal_coef * self.value_terminal_energy(h, returns, old_values, args)

        return self.settle_states(self.critic_pc, x, initial_states, args, terminal)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)
    if args.torch_compile:
        try:
            import torch._dynamo as dynamo

            dynamo.config.cache_size_limit = 256
        except Exception:
            pass
        if args.torch_compile_disable_cudagraphs:
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
    head_optimizer = optim.Adam(agent.head_parameters(), lr=args.learning_rate, eps=1e-5)
    local_optimizer = optim.Adam(agent.local_parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            head_optimizer.param_groups[0]["lr"] = lrnow
            local_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, action_z, logprob, _, value = agent.get_action_and_value(next_obs, args)
                values[step] = value.flatten()
            actions[step] = action
            action_zs[step] = action_z
            logprobs[step] = logprob

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

        with torch.no_grad():
            next_value = agent.get_value(next_obs, args).reshape(1, -1)
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

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_action_zs = action_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        actor_energy_drop = []
        critic_energy_drop = []
        actor_local_loss_value = 0.0
        critic_local_loss_value = 0.0
        pg_loss = torch.zeros((), device=device)
        v_loss = torch.zeros((), device=device)
        entropy_loss = torch.zeros((), device=device)
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                actor_states, actor_energy = agent.infer_actor_update_states(
                    b_obs[mb_inds],
                    b_action_zs[mb_inds],
                    b_logprobs[mb_inds],
                    mb_advantages,
                    args,
                )
                critic_states, critic_energy = agent.infer_critic_update_states(
                    b_obs[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                    args,
                )
                actor_energy_drop.append((actor_energy[0] - actor_energy[1]).item())
                critic_energy_drop.append((critic_energy[0] - critic_energy[1]).item())

                deploy_actor_states = agent.deployment_states(agent.actor_pc, b_obs[mb_inds], args)
                deploy_critic_states = agent.deployment_states(agent.critic_pc, b_obs[mb_inds], args)

                dist = agent.actor_dist_from_state(deploy_actor_states[-1])
                mb_action_zs = b_action_zs[mb_inds].detach().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                newlogprob = dist.log_prob(mb_action_zs).sum(1)
                entropy = dist.entropy().sum(1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                pg_loss1 = -mb_advantages.detach() * ratio
                pg_loss2 = -mb_advantages.detach() * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = agent.value_from_state(deploy_critic_states[-1]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds].detach()) ** 2
                    v_clipped = b_values[mb_inds].detach() + torch.clamp(
                        newvalue - b_values[mb_inds].detach(),
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds].detach()) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds].detach()) ** 2).mean()

                entropy_loss = entropy.mean()
                terminal_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                actor_local_loss = args.pc_local_energy_coef * agent.actor_pc.local_learning_loss(
                    b_obs[mb_inds],
                    actor_states,
                    args,
                )
                critic_local_loss = args.pc_local_energy_coef * agent.critic_pc.local_learning_loss(
                    b_obs[mb_inds],
                    critic_states,
                    args,
                )
                actor_local_loss_value = actor_local_loss.item()
                critic_local_loss_value = critic_local_loss.item()

                head_optimizer.zero_grad()
                terminal_loss.backward()
                nn.utils.clip_grad_norm_(agent.head_parameters(), args.max_grad_norm)
                head_optimizer.step()

                local_optimizer.zero_grad()
                (actor_local_loss + critic_local_loss).backward()
                nn.utils.clip_grad_norm_(agent.local_parameters(), args.max_grad_norm)
                local_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", head_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("pc/actor_local_loss", actor_local_loss_value, global_step)
        writer.add_scalar("pc/critic_local_loss", critic_local_loss_value, global_step)
        writer.add_scalar("pc/actor_energy_drop", np.mean(actor_energy_drop), global_step)
        writer.add_scalar("pc/critic_energy_drop", np.mean(critic_energy_drop), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(clean_compiled_state_dict(agent), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
