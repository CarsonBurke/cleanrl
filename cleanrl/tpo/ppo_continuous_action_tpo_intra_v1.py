# Intra-trajectory Target Policy Optimization (TPO) for continuous control. v1.
#
# Method. TPO (Kaddour 2026) replaces the policy gradient with a closed-form
# target distribution q ∝ p_old · exp(u/eta) and fits the policy to it by cross
# entropy, so the loss gradient is (p - q) and vanishes exactly at the target.
# The paper's group form compares whole completions; this file implements the
# *intra-trajectory* form: every timestep is its own TPO problem over the binary
# outcome "the executed action vs. the entire rest of the action space", which
# is the continuous generalisation of the paper's Appendix-C one-sampled-action
# construction. One behaviour action per state, no counterfactual rollouts, no
# candidate-Q estimates, no comparison of differently sized sequences.
#
# Derivation. Give the executed action a_t a cell of width delta in each of the
# d action dimensions, so it carries genuine probability mass
#   p_old = pi_old(a_t|s_t) · delta^d,     rest = 1 - p_old.
# Utility u_t lands on the executed outcome only, so the anchored TPO target is
#   q_t = sigmoid(logit(p_old) + u_t/eta)
# and the fit is binary cross entropy against p_theta = pi_theta(a_t|s_t)·delta^d.
# For any physically sensible cell p_old, p_theta << 1, and BCE reduces exactly to
#   grad L_t = delta^d · pi_old(a_t) · (r_t - w_t) · grad log pi_theta(a_t|s_t),
#   r_t = pi_theta/pi_old   (ratio),      w_t = exp(u_t/eta)   (target ratio).
# The per-sample prefactor delta^d·pi_old is the only place the cell enters;
# dropping it weights every trajectory position equally — the continuous analogue
# of "mean over tokens, then mean over trajectories" — and leaves a delta-free
# objective with the same gradient:
#   L_t = w_t · d(r_t / w_t),   d(x) = x - log x - 1 >= 0
#       = r_t - w_t·log r_t - w_t + w_t·log w_t
#   grad L_t = (r_t - w_t) · grad log pi_theta(a_t|s_t).
#
# Why this should beat the PPO surrogate.
#  - True fixed point. L_t is convex in log r_t and minimised uniquely at
#    r_t = exp(u_t/eta). PPO's clipped surrogate has no fixed point: it pushes
#    monotonically until clipping switches the gradient off. Ten epochs over the
#    same rollout therefore *converge* here instead of drifting, and no clipping
#    coefficient is needed.
#  - Exact neutrality. u_t = 0 gives w_t = 1, so the gradient is exactly zero at
#    r_t = 1 — states the critic cannot separate contribute nothing. Whitening
#    advantages over the whole rollout makes a variance-free batch actor-neutral,
#    the continuous analogue of the zero-reward-variance prompt-group safeguard.
#  - Self-correcting trust region. grad ∝ (r_t - w_t) is bounded by w_t as
#    r_t -> 0 (suppression cannot explode) and grows linearly in r_t, so runaway
#    probability mass is pulled back harder the further it runs.
#  - Advantage-proportional step size. The target ratio is exp(u_t/eta) rather
#    than a uniform +/-clip_coef band, so exceptional actions may move much
#    further than mediocre ones within one update.
#
# Advantages are whitened over the full rollout (not per minibatch) so that the
# target q_t is a single fixed quantity for the entire update, as TPO requires;
# per-minibatch whitening would re-randomise the target every epoch. eta is the
# trust-region knob: with whitened u clipped at +/-utility_clip, the largest
# demanded probability change is exp(utility_clip/eta).
#
# Hypothesis: replacing PPO's clipped surrogate with the intra-trajectory TPO
# target, keeping GAE/critic/architecture identical to the baseline, improves
# HalfCheetah-v4 return by making multi-epoch reuse of each rollout converge to
# a well-defined target instead of relying on clipping to halt drift.

import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    eta: float = 4.0
    """TPO temperature; the target ratio is exp(utility / eta), so smaller eta
    demands larger probability changes per update"""
    utility_clip: float = 3.0
    """clip whitened utilities to +/- this many standard deviations, bounding the
    target ratio to exp(utility_clip / eta)"""
    logratio_guard: float = 20.0
    """numerical overflow guard on the log ratio before exponentiating; this is
    not a trust region and should essentially never bind"""
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


def tpo_utility(advantages: torch.Tensor, utility_clip: float) -> torch.Tensor:
    """Whiten advantages into TPO utilities (the paper's `tpo_skill`), then clip.

    A batch with no advantage variance maps to all-zero utility, which makes the
    actor exactly neutral on it.
    """
    centered = advantages - advantages.mean()
    std = centered.std()
    utility = torch.where(std > 1e-6, centered / std.clamp_min(1e-6), centered)
    if utility_clip > 0.0:
        utility = utility.clamp(-utility_clip, utility_clip)
    return utility


def tpo_intra_loss(
    logratio: torch.Tensor,
    utility: torch.Tensor,
    eta: float,
    logratio_guard: float,
) -> torch.Tensor:
    """Per-sample intra-trajectory TPO loss with gradient (ratio - target) * dlogpi.

    ``L = w * (r/w - log(r/w) - 1)`` for ``r = exp(logratio)`` and target ratio
    ``w = exp(utility / eta)``. Non-negative, convex in ``logratio``, and zero
    exactly at ``r = w``; ``utility = 0`` therefore gives zero gradient at ``r = 1``.
    """
    log_target_ratio = utility.detach() / eta
    target_ratio = log_target_ratio.exp()
    safe_logratio = logratio.clamp(-logratio_guard, logratio_guard)
    ratio = safe_logratio.exp()
    return (
        ratio
        - target_ratio * safe_logratio
        - target_ratio
        + target_ratio * log_target_ratio
    )


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


class Agent(nn.Module):
    """Unchanged from the PPO baseline: only the policy objective differs."""

    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


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
                action, logprob, _, value = action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
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
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                _, newlogprob, entropy, newvalue = action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    movefracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]

                if args.utility_scope == "batch":
                    mb_utilities = b_utilities[mb_inds]
                else:
                    mb_utilities = tpo_utility(b_advantages[mb_inds], args.utility_clip)

                # Policy loss: fit the executed action's probability to the
                # anchored TPO target pi_old * exp(u / eta). No clipping.
                pg_loss = tpo_intra_loss(logratio, mb_utilities, args.eta, args.logratio_guard).mean()

                with torch.no_grad():
                    # |ratio - target| is the TPO fit error; it should shrink
                    # across epochs because the target does not move.
                    residuals += [(ratio - (mb_utilities / args.eta).exp()).abs().mean().item()]

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
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/movefrac", np.mean(movefracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("tpo/residual_first_epoch", residuals[0], global_step)
        writer.add_scalar("tpo/residual_last_epoch", residuals[-1], global_step)
        writer.add_scalar("tpo/target_ratio_max", float(np.exp(args.utility_clip / args.eta)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
