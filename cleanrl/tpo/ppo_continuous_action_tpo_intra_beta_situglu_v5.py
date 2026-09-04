# Intra-trajectory TPO, Beta policy, SiTU-GLU trunks, scalar critic. v5.
#
# v5 is v4 with the HL-Gauss categorical critic removed and nothing else changed:
# same TPO objective, same Beta policy, same SiTU-GLU trunks with learned cap
# allocation, same separate per-network reduce-overhead compilation, same eta.
# The critic head is a single scalar with PPO's clipped value loss restored, as
# in v2. This isolates the distributional critic as a single variable.
#
# Motivation. v4 (SiTU-GLU + HL-Gauss, eta=4) trails the plain ReluSq + scalar
# critic run badly on return, yet its explained variance is *higher* (~0.83 vs
# ~0.67). A better-fitting critic paired with worse control points away from the
# critic head as the cause, so this run removes it to confirm that directly. A
# companion run varies eta alone on the unmodified v4 file.
#
# The TPO objective is unchanged. Every timestep is its own TPO problem over
# "the executed action vs. the entire rest of the action space". Give the
# executed action a cell of width delta per dimension so it carries genuine mass
# p_old = pi_old(a_t|s_t)*delta^d; the anchored target is
# q_t = sigmoid(logit(p_old) + u_t/eta), fitted by binary cross entropy. For any
# sensible cell p_old, p_theta << 1 and the BCE reduces exactly to a delta-free
#   L_t = w_t*(x - log x - 1),  x = r_t/w_t,  w_t = exp(u_t/eta)
#   grad L_t = (r_t - w_t)*grad log pi_theta(a_t|s_t),
# whose unique fixed point is r_t = w_t. u_t is the rollout-whitened advantage,
# so eta is how many advantage standard deviations demand an e-fold change in
# the executed action's probability. Only exp() is guarded, by linearisation
# rather than clamping, so suppressed samples keep their restoring gradient.
#
# Hypothesis: removing HL-Gauss changes little, leaving eta (and secondarily the
# SiTU-GLU trunk) as the explanation for v4's shortfall.

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
# E[SiTU(g,u)^2] at the inherited sqrt(2) gate/up gain under isotropic
# unit-variance input (Gauss-Hermite quadrature), from sf_vlam_v8.
SITU_SECOND_MOMENT = 1.2630450818573506
# Second moment of the ReluSq branch this is scale-matched against:
# 2 * E[ReLU(N(0,2))^4] = 12.
RELUSQ_SECOND_MOMENT = 12.0
# v4's HL-Gauss head width. v5 advances the RNG by exactly this head's draws so
# the actor below initialises bit-identically to v4's, making the two runs
# seed-paired and leaving the critic as the only difference between them.
V4_HLGAUSS_HEAD_WIDTH = 511


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
    compile: bool = True
    """compile the actor and critic networks separately with torch.compile"""
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
    hidden_dim: int = 64
    """width of the actor and critic trunks"""
    utility_scope: Literal["batch", "minibatch"] = "batch"
    """whiten the GAE advantages into TPO utilities over the whole rollout
    (keeps the target fixed for the entire update) or per minibatch"""
    eta: float = 4.0
    """TPO temperature: how many advantage standard deviations are needed to
    demand an e-fold change in the executed action's probability. The target
    ratio is exp(utility / eta), so smaller eta demands larger changes"""
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


# --- TPO objective (unchanged from v2) ---------------------------------------


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


def tpo_intra_loss(
    logratio: torch.Tensor,
    utility: torch.Tensor,
    eta: float,
    logratio_guard: float,
) -> torch.Tensor:
    """Per-sample intra-trajectory TPO loss with gradient (ratio - target).

    ``L = w * (x - log x - 1)`` for ``x = r/w``, ``r = exp(logratio)`` and target
    ``w = exp(utility / eta)``. Non-negative, convex in ``logratio``, zero exactly
    at ``r = w``. Only ``exp`` is guarded, and by linearisation rather than
    clamping, so a heavily suppressed sample keeps its full restoring gradient.
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


# --- SiTU-GLU ----------------------------------------------------------------


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if getattr(layer, "bias", None) is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def situ_glu(gate, up, beta_gate, beta_up):
    """SiTU-GLU product with a fixed cap product and learned cap allocation."""
    # The invariant product is factored explicitly rather than multiplying
    # separately rounded reciprocal exponentials back together.
    return (
        100.0
        * torch.tanh(gate / beta_gate)
        * torch.sigmoid(gate)
        * torch.tanh(up / beta_up)
    )


class SiTUGLUBranch(nn.Module):
    """SiTU-GLU FFN with learned cap allocation, scale-matched to a ReluSq stage."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # A zero-centered log-delta keeps the official allocation bit-exact at
        # initialization while moving reciprocal scale between the two caps, so
        # the cap *product* (and hence the activation range) stays invariant.
        self.log_beta_delta = nn.Parameter(torch.tensor(0.0))
        # Official SiTU-GLU uses three bias-free matrices; M = round(2(H+1)/3) is
        # the parameter-matched gated width against two biased H->H linears.
        self.gated_dim = max(1, round(2.0 * (out_dim + 1) / 3.0))
        # Preserve the initial output scale of the ReluSq stage being replaced.
        # A rectangular orthogonal M->H down projection contributes M/H on
        # average, giving the gain below; the differing Jacobian geometry is the
        # thing actually under test.
        down_gain = np.sqrt(
            RELUSQ_SECOND_MOMENT * out_dim / (self.gated_dim * SITU_SECOND_MOMENT)
        )
        self.gate = layer_init(nn.Linear(in_dim, self.gated_dim, bias=False))
        self.up = layer_init(nn.Linear(in_dim, self.gated_dim, bias=False))
        self.down = layer_init(nn.Linear(self.gated_dim, out_dim, bias=False), std=down_gain)

    def beta_gate(self):
        return 4.0 * torch.exp(self.log_beta_delta)

    def beta_up(self):
        return 25.0 * torch.exp(-self.log_beta_delta)

    def forward(self, x):
        return self.down(situ_glu(self.gate(x), self.up(x), self.beta_gate(), self.beta_up()))


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


class Agent(nn.Module):
    """Beta policy on the native action box with a scalar critic.

    The actor and critic are exposed as two separate forward methods so each
    network can be compiled independently; all distribution math stays in eager.
    """

    def __init__(self, envs, hidden_dim=64):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        h = hidden_dim
        self.critic_trunk = nn.Sequential(SiTUGLUBranch(obs_dim, h), SiTUGLUBranch(h, h))
        # Consume the draws v4's 511-bucket head would have taken at this point,
        # so the actor is seed-paired with v4 and the ablation is critic-only.
        _ = layer_init(nn.Linear(h, V4_HLGAUSS_HEAD_WIDTH), std=0.1)
        self.actor_trunk = nn.Sequential(SiTUGLUBranch(obs_dim, h), SiTUGLUBranch(h, h))
        self.actor_alpha = layer_init(nn.Linear(h, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(h, action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(h, 1), std=1.0)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def policy_params(self, x):
        """Compiled actor network: observations -> Beta concentrations."""
        h = self.actor_trunk(x)
        # alpha, beta >= 1 keeps the density unimodal and finite at the edges.
        return 1.0 + F.softplus(self.actor_alpha(h)), 1.0 + F.softplus(self.actor_beta(h))

    def critic_value(self, x):
        """Compiled critic network: observations -> scalar state value."""
        return self.critic_head(self.critic_trunk(x))

    def z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z


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

    agent = Agent(envs, args.hidden_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    situ_branches = [m for m in agent.modules() if isinstance(m, SiTUGLUBranch)]

    # Each network is compiled on its own. The Beta sampling and log-prob math is
    # deliberately left in eager: RNG inside a captured cudagraph is a hazard, and
    # the distribution ops are cheap relative to the trunks.
    policy_params = agent.policy_params
    critic_value = agent.critic_value
    if args.compile:
        policy_params = torch.compile(policy_params, mode=args.compile_mode, dynamic=False)
        critic_value = torch.compile(critic_value, mode=args.compile_mode, dynamic=False)
        print(f"compiled actor and critic networks separately (mode={args.compile_mode!r})")

    def step_begin():
        # Once per iteration, before the first compiled call -- not between the
        # actor and critic calls, whose activations must both survive to backward.
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

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
                step_begin()
                alpha, beta = policy_params(next_obs)
                dist = Beta(alpha, beta)
                z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                logprob = dist.log_prob(z).sum(1)
                action = agent.z_to_action(z)
                values[step] = critic_value(next_obs).flatten()
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
            step_begin()
            next_value = critic_value(next_obs).reshape(1, -1)
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

                step_begin()
                alpha, beta = policy_params(b_obs[mb_inds])
                dist = Beta(alpha, beta)
                mb_z = b_zs[mb_inds]
                newlogprob = dist.log_prob(mb_z).sum(1)
                entropy = dist.entropy().sum(1)
                newvalue = critic_value(b_obs[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.clamp_max(args.logratio_guard).exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    movefracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]
                    concentrations += [(alpha + beta).mean().item()]

                if args.utility_scope == "batch":
                    mb_utilities = b_utilities[mb_inds]
                else:
                    mb_utilities = tpo_utility(b_advantages[mb_inds], args.utility_clip)

                # Policy loss: fit the executed action's probability to the
                # anchored TPO target pi_old * exp(u / eta). No clipping.
                pg_loss = tpo_intra_loss(logratio, mb_utilities, args.eta, args.logratio_guard).mean()

                with torch.no_grad():
                    # |ratio - target| is the TPO fit error; its max is the
                    # per-sample gradient coefficient, so a large value means one
                    # sample is dominating the minibatch direction.
                    coefficient = ratio - (mb_utilities / args.eta).exp()
                    residuals += [coefficient.abs().mean().item()]
                    coef_maxes += [coefficient.abs().max().item()]

                # Value loss: PPO's clipped scalar regression, as in v2.
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.vloss_clip_coef,
                        args.vloss_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
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
        with torch.no_grad():
            beta_gates = torch.stack([b.beta_gate() for b in situ_branches])
            beta_ups = torch.stack([b.beta_up() for b in situ_branches])
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
        writer.add_scalar("situ/beta_gate_mean", beta_gates.mean().item(), global_step)
        writer.add_scalar("situ/beta_gate_min", beta_gates.min().item(), global_step)
        writer.add_scalar("situ/beta_gate_max", beta_gates.max().item(), global_step)
        writer.add_scalar("situ/beta_up_mean", beta_ups.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
