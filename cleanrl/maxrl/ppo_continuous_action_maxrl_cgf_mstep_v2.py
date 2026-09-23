# MaxRL-CGF M-step v2: fit the success posterior instead of pushing a clipped surrogate.
#
# v1 (see ppo_continuous_action_maxrl_cgf_v1.py) established the objective: success
# likelihood e^{beta G}, LINEX soft-value critic psi = (1/b) log E[e^{b G}], soft GAE A,
# posterior weight w = e^{bA} with E[w | s] = 1 at the critic's fixed point. Measured
# there: the soft critic alone beats PPO, but e^{bA} inside PPO's clipped surrogate learns
# fastest early and then plateaus. Diagnosis: the clip saturates the few large-weight
# samples after about one step, and the bulk, at -1 after centering, keeps pushing density
# down. That is an artifact of the surrogate, not of the objective.
#
# MaxRL's gradient is grad log p = E[score | success], the first-order step of an EM
# M-step: maximize E_q[log pi], with q(a|s) proportional to pi_old(a|s) E[w | s, a].
# Here that fit is taken directly:
#     maximize  mean_i  w_i / mean(w) * log pi(a_i | s_i)
# under two MPO-style trust regions on the Beta head, one per statistic. The mean
# m = a/(a+b) and concentration c = a+b are fitted through Beta(m_new, c_old) and
# Beta(m_old, c_new), with KL(old || each) <= eps_mean and eps_conc. Each constraint is
# enforced by a Lagrange multiplier learned by dual descent. The M-step has a bounded optimum,
# and it sets the policy's spread to the spread of successful actions, under its own
# budget, instead of collapsing it through a shared clip. Critic, temperature
# (beta = kappa / std A) and separate gradient clipping are as in v1.
import math
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
    """value-function clipping range"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    maxrl_kappa: float = 1.0
    """dimensionless temperature: beta = kappa / std(soft advantage) per batch"""
    eps_mean: float = 0.01
    """per-iteration KL budget for the Beta mean (PPO spends ~0.008 in total)"""
    eps_conc: float = 0.001
    """per-iteration KL budget for the Beta concentration"""
    dual_lr: float = 0.01
    """Adam step size for the log Lagrange multipliers"""

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


def linex_loss(prediction, target, beta):
    """(e^{bu} - bu - 1)/b^2, u = target - prediction; minimizer is (1/b) log E[e^{b target}]."""
    scaled = beta * (target - prediction)
    return (torch.expm1(scaled) - scaled) / (beta * beta)


def posterior_weights(soft_advantages, beta):
    """e^{bA} over its batch mean: a global scale, so relative weights across states are kept."""
    log_w = beta * soft_advantages
    return torch.exp(log_w - (torch.logsumexp(log_w, 0) - math.log(log_w.numel())))


def posterior_diagnostics(soft_advantages, beta):
    """Batch statistics of w = e^{bA} in log space: calibration (mean ~ 1), tail mass, ESS."""
    log_w = beta * soft_advantages
    log_n = math.log(log_w.numel())
    log_total = torch.logsumexp(log_w, 0)
    top = torch.topk(log_w, max(1, log_w.numel() // 10), sorted=False).values
    return {
        "maxrl/log_w_mean": log_total - log_n,
        "maxrl/log_w_max": log_w.max(),
        "maxrl/ess_frac": torch.exp(2 * log_total - log_n - torch.logsumexp(2 * log_w, 0)),
        "maxrl/top10_mass": torch.exp(torch.logsumexp(top, 0) - log_total),
        "maxrl/soft_adv_mean": soft_advantages.mean(),
        "maxrl/soft_adv_std": soft_advantages.std(),
        "maxrl/beta": beta,
    }


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


def beta_kl(alpha_p, beta_p, alpha_q, beta_q):
    """Closed-form KL(Beta(alpha_p, beta_p) || Beta(alpha_q, beta_q)); traceable, unlike kl_divergence."""
    total_p, total_q = alpha_p + beta_p, alpha_q + beta_q
    log_norm = (torch.lgamma(alpha_q) + torch.lgamma(beta_q) - torch.lgamma(total_q)
                - torch.lgamma(alpha_p) - torch.lgamma(beta_p) + torch.lgamma(total_p))
    return (log_norm + (alpha_p - alpha_q) * torch.digamma(alpha_p) + (beta_p - beta_q) * torch.digamma(beta_p)
            + (total_q - total_p) * torch.digamma(total_p))


def mean_concentration(alpha, beta):
    concentration = alpha + beta
    return alpha / concentration, concentration


def mstep_loss(agent, observations, native_actions, old_alpha, old_beta, weights, returns,
               old_values, temperature, log_multipliers, args):
    """Decoupled weighted-ML fit to the success posterior, dual-descent KL budgets, LINEX critic."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
    mean, concentration = mean_concentration(alpha, beta)
    old_mean, old_concentration = mean_concentration(old_alpha, old_beta)
    mean_alpha, mean_beta = mean * old_concentration, (1.0 - mean) * old_concentration
    conc_alpha, conc_beta = old_mean * concentration, (1.0 - old_mean) * concentration
    mean_step = Beta(mean_alpha, mean_beta, validate_args=False)
    concentration_step = Beta(conc_alpha, conc_beta, validate_args=False)
    log_likelihood = (mean_step.log_prob(native_actions) + concentration_step.log_prob(native_actions)).sum(-1)
    fit_loss = -(weights * log_likelihood).mean()
    kl = torch.stack((beta_kl(old_alpha, old_beta, mean_alpha, mean_beta).sum(-1).mean(),
                      beta_kl(old_alpha, old_beta, conc_alpha, conc_beta).sum(-1).mean()))
    multipliers = log_multipliers.exp()
    budgets = kl.new_tensor((args.eps_mean, args.eps_conc))
    policy_loss = fit_loss + (multipliers.detach() * kl).sum()
    dual_loss = (multipliers * (budgets - kl.detach())).sum()
    newvalue = newvalue.view(-1)
    v_loss_unclipped = linex_loss(newvalue, returns, temperature)
    if args.clip_vloss:
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = torch.max(v_loss_unclipped, linex_loss(v_clipped, returns, temperature)).mean()
    else:
        v_loss = v_loss_unclipped.mean()
    entropy = (Beta(alpha, beta, validate_args=False).entropy() + agent.log_action_scale).sum(-1).mean()
    loss = policy_loss + v_loss * args.vf_coef + dual_loss
    metrics = torch.cat((torch.stack((fit_loss.detach(), v_loss.detach(), entropy.detach())),
                         kl.detach(), multipliers.detach()))
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
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
    if min(args.maxrl_kappa, args.eps_mean, args.eps_conc, args.dual_lr) <= 0:
        raise ValueError("maxrl_kappa, KL budgets and dual_lr must be positive")
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
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror; "
                        f"MaxRL-CGF M-step kappa={args.maxrl_kappa} eps_mean={args.eps_mean} eps_conc={args.eps_conc}")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
        # Lagrange multipliers (mean, concentration) for the KL budgets; dual ascent in log space.
        log_multipliers = nn.Parameter(torch.zeros(2, device=device))
        dual_optimizer = optim.Adam([log_multipliers], lr=args.dual_lr)
        value_model = agent.get_value

        def rollout_statistics(observations):
            """Old Beta parameters and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), alpha, beta

        def loss_model(observations, native, old_alpha, old_beta, weights, returns, old_values, temperature):
            return mstep_loss(agent, observations, native, old_alpha, old_beta, weights, returns,
                              old_values, temperature, log_multipliers, args)

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
        update_metrics = torch.empty((max_updates, 9), device=device)
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
                b_values, b_alpha, b_beta = rollout_statistics(b_obs)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                # Soft GAE: the unchanged recursion over psi.
                soft_advantages = advantages.flatten()
                # Temperature in units of this batch's soft-advantage spread, so kappa keeps
                # its meaning as the reward normalizer and the critic rescale returns.
                beta = args.maxrl_kappa / (soft_advantages.std() + 1e-8)
                b_weights = posterior_weights(soft_advantages, beta)
                b_returns = returns.flatten().clone()
                posterior = posterior_diagnostics(soft_advantages, beta)
            updates = 0
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_alpha[indices], b_beta[indices],
                            b_weights[indices], b_returns[indices], b_values[indices], beta,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        dual_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        # Separate norms: LINEX gradients are exponential in the critic's
                        # underestimate and must not shrink the actor's step.
                        actor_norm = nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        critic_norm = nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        optimizer.step()
                        dual_optimizer.step()
                        update_metrics[updates, :7].copy_(metrics)
                        update_metrics[updates, 7] = actor_norm
                        update_metrics[updates, 8] = critic_norm
                        updates += 1

            last = update_metrics[updates - 1]
            logged = gather_metrics({
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "mstep/kl_mean": last[3], "mstep/kl_conc": last[4],
                "mstep/multiplier_mean": last[5], "mstep/multiplier_conc": last[6],
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "losses/actor_grad_norm": update_metrics[:updates, 7].mean(),
                "losses/critic_grad_norm": update_metrics[:updates, 8].mean(),
                **posterior,
            })
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
