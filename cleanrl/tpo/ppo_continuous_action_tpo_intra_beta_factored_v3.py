# Intra-trajectory TPO with a Beta policy and a per-coordinate factored target. v3.
#
# v3 is v2 plus exactly one change: the TPO problem is posed independently per
# action coordinate instead of once over the joint action. Everything else — the
# Beta policy, the whitened utilities, eta, the guard, the critic, the loop — is
# identical to ppo_continuous_action_tpo_intra_beta_v2.py, so this is a clean
# ablation of the factorisation alone.
#
# Why factor. v2 fits the *joint* ratio r_t = prod_i pi_theta(a_ti)/pi_old(a_ti).
# Two problems follow from the product:
#  - Variance. log r_t is a sum of d independent per-coordinate terms, so its
#    spread grows like sqrt(d) and the loss, being exponential in log r_t, has
#    gradient variance growing like exp(Var). Fine at d=6, bad at d >> 20.
#  - Credit. The joint gradient (r_t - w_t) multiplies grad log pi for *every*
#    coordinate equally, so a coordinate that has already reached its share of
#    the target keeps being pushed because the other coordinates have not.
# Factoring gives d independent binary TPO problems per timestep, each with an
# O(1) log-ratio and its own stopping condition. This is the same rule the paper
# applies along the sequence axis (mean over tokens rather than a product over
# tokens) applied along the action-coordinate axis: in an LLM each position is a
# single categorical draw and the sequence is the many-factor product, whereas in
# continuous control a single timestep is already a d-factor product.
#
# Matching the trust region. The scalar advantage gives no per-coordinate credit,
# so every coordinate receives the same utility u_t. To make v3 differ from v2
# *only* in the factorisation, both the fixed point and the linearised step size
# are matched:
#   per-coordinate target   w_dim = exp(u_t / (eta * d))
#   loss                    L_t = d * sum_i w_dim * dist(r_ti / w_dim)
#   gradient                dL_t/dlogratio_ti = d * (r_ti - w_dim)
# so the joint fixed point is prod_i w_dim = exp(u_t / eta), identical to v2, and
# near r=1 the per-coordinate gradient is d*(-u_t/(eta*d)) = -u_t/eta, also
# identical to v2. Without the d prefactor the effective actor step would be d
# times smaller and the comparison would confound step size with factorisation.
#
# Hypothesis: per-coordinate targets improve HalfCheetah-v4 (d=6) modestly by
# better credit assignment, and the gap should widen with action dimension.

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

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
    utility_scope: Literal["batch", "minibatch"] = "batch"
    """whiten the GAE advantages into TPO utilities over the whole rollout
    (keeps the target fixed for the entire update) or per minibatch"""
    eta: float = 2.0
    """TPO temperature: how many advantage standard deviations are needed to
    demand an e-fold change in the *joint* action probability. The per-coordinate
    target is exp(utility / (eta * action_dim))"""
    utility_clip: float | None = 3.0
    """clip whitened utilities to +/- this many standard deviations, bounding the
    joint target ratio to exp(utility_clip / eta); None disables clipping"""
    logratio_guard: float = 20.0
    """per-coordinate log-ratio above which exp() is linearised, purely to keep
    the loss and its gradient finite. Unlike a clip this preserves a monotone,
    non-zero gradient, and never affects the restoring gradient below it"""
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
    exactly neutral on it.
    """
    centered = advantages - advantages.mean()
    std = centered.std(unbiased=False)
    utility = torch.where(std > 1e-6, centered / std.clamp_min(1e-6), centered)
    if utility_clip is not None:
        utility = utility.clamp(-utility_clip, utility_clip)
    return utility


def tpo_intra_loss_factored(
    logratio: torch.Tensor,
    utility: torch.Tensor,
    eta: float,
    logratio_guard: float,
) -> torch.Tensor:
    """Per-sample TPO loss over d independent per-coordinate binary problems.

    ``logratio`` is ``[batch, action_dim]`` of per-coordinate log ratios and
    ``utility`` is ``[batch]``. Each coordinate is fitted to the same target
    ``w_dim = exp(utility / (eta * action_dim))``, and the result is scaled by
    ``action_dim`` so that both the joint fixed point ``exp(utility / eta)`` and
    the linearised per-coordinate gradient ``-utility / eta`` match the joint
    formulation in v2.

    As in v2 only ``exp`` is guarded, and by linearisation rather than clamping,
    so a heavily suppressed coordinate keeps its full restoring gradient.
    """
    action_dim = logratio.shape[-1]
    log_target_ratio = utility.detach().unsqueeze(-1) / (eta * action_dim)
    target_ratio = log_target_ratio.exp()
    guard_ratio = math.exp(logratio_guard)
    ratio = torch.where(
        logratio > logratio_guard,
        guard_ratio * (logratio - logratio_guard + 1.0),
        logratio.clamp_max(logratio_guard).exp(),
    )
    per_coordinate = (
        ratio - target_ratio * logratio - target_ratio + target_ratio * log_target_ratio
    )
    return action_dim * per_coordinate.sum(-1)


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
    """Beta policy on the native action box; identical to v2."""

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
        """Return per-coordinate log-probs; the factored loss needs them unsummed."""
        dist = self._dist(x)
        if z is None:
            z = dist.sample()
        z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        logprob = dist.log_prob(z)  # [batch, action_dim], deliberately not summed
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

    # ALGO Logic: Storage setup. logprobs are per-coordinate for the factored loss.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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
        b_logprobs = logprobs.reshape((-1,) + envs.single_action_space.shape)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ALGO LOGIC: the TPO target is fixed for the whole update, so utilities
        # are whitened once over the rollout rather than per minibatch.
        if args.utility_scope == "batch":
            b_utilities = tpo_utility(b_advantages, args.utility_clip)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        residuals = []
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
                logratio = newlogprob - b_logprobs[mb_inds]  # [minibatch, action_dim]
                ratio = logratio.clamp_max(args.logratio_guard).exp()

                with torch.no_grad():
                    # KL diagnostics stay on the joint action for comparability
                    # with v2 and the PPO baseline.
                    joint_logratio = logratio.sum(-1)
                    joint_ratio = joint_logratio.clamp_max(args.logratio_guard).exp()
                    old_approx_kl = (-joint_logratio).mean()
                    approx_kl = ((joint_ratio - 1) - joint_logratio).mean()
                    movefracs += [((joint_ratio - 1.0).abs() > 0.2).float().mean().item()]
                    concentrations += [concentration.item()]

                if args.utility_scope == "batch":
                    mb_utilities = b_utilities[mb_inds]
                else:
                    mb_utilities = tpo_utility(b_advantages[mb_inds], args.utility_clip)

                # Policy loss: d independent per-coordinate TPO fits. No clipping.
                pg_loss = tpo_intra_loss_factored(
                    logratio, mb_utilities, args.eta, args.logratio_guard
                ).mean()

                with torch.no_grad():
                    # Per-coordinate fit error against the per-coordinate target.
                    action_dim = logratio.shape[-1]
                    coefficient = ratio - (mb_utilities.unsqueeze(-1) / (args.eta * action_dim)).exp()
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
            float(np.exp((args.utility_clip or np.inf) / args.eta)),
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
