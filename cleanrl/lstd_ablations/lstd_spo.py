"""LSTD + Simple Policy Optimization (SPO) — Xie et al. 2024 (arXiv:2401.16025)

Fork of lstd_linear_tanh. Replaces PPO's hard clipped surrogate with SPO's
quadratic-penalty objective:

    L_SPO = r_t * A_t  -  |A_t| / (2ε) * (r_t - 1)²

Key properties vs PPO clip:
1. **Never-zero gradients**: PPO zeros gradients when ratio exits [1-ε, 1+ε],
   creating flat regions. SPO's smooth penalty always provides gradient signal.
   This is especially relevant for SDE noise params which need continuous
   gradient feedback to learn good noise schedules.
2. **Advantage-scaled penalty**: high-|A| transitions get stronger ratio
   constraints, preventing destructive large updates on the most impactful
   samples. Low-|A| transitions get softer penalties, allowing larger steps
   where they're less harmful.
3. **ε-aligned**: optimal ratio r* = 1 + sign(A)*ε, same target as PPO but
   reached via smooth optimization rather than piecewise clipping.

Extensions over vanilla SPO:
1. **Asymmetric ε**: uses clip_coef_high (0.28) for positive advantages and
   clip_coef_low (0.2) for negative, matching our PPO baseline's asymmetry.
   Weaker penalty on the upside (coef 1.79 vs 2.5) allows larger beneficial
   ratio changes.
2. **Learned bias**: optional learnable scalar `spo_bias` shifts the penalty
   center from r=1 to r=1+bias. The gradient pushes bias toward mean(r-1),
   adaptively centering the trust region where the policy actually moves.
   This helps early training when the policy needs large initial changes.

Value loss: kept with PPO-style clipping (SPO paper only changes policy loss).
"""
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
SDE_EPS = 1e-6
SDE_PRESCALE = 1.5  # divisor before tanh, controls saturation


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
    wandb_entity: str = None
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
    num_envs: int = 1
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
    clip_coef_low: float = 0.2
    """initial SPO ε for negative advantages; also used for value loss clipping"""
    clip_coef_high: float = 0.28
    """initial SPO ε for positive advantages; also used for value loss clipping"""
    spo_target_dev: float = 0.005
    """target for E[|A|*(r-1)^2]; closed-loop advantage-weighted ratio constraint"""
    spo_eps_lr: float = 3e-4
    """learning rate for the dual variables log_eps_high / log_eps_low"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""

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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        hidden_dim = 64

        # Actor backbone: obs -> hidden features
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_norm2 = RMSNorm(hidden_dim)

        # Actor output head -- init std=1.0 so weights are meaningful for noise modulation
        self.actor_out = layer_init(nn.Linear(hidden_dim, act_dim), std=1.0)
        # Separate learnable scale for the mean (starts small -> near-zero initial actions)
        self.mean_scale = nn.Parameter(torch.tensor(0.01))

        # SDE noise with learned log_std_param (trading bot approach)
        self.sde_fc = layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.sde_norm = RMSNorm(hidden_dim)
        self.sde_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        # Learned per-element noise magnitude: log_std = (param + LOG_STD_INIT).clamp(min, max)
        self.log_std_param = nn.Parameter(torch.zeros(hidden_dim, act_dim))


        # Critic backbone (separate from actor)
        self.critic_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_norm2 = RMSNorm(hidden_dim)

        # Scalar value head
        self.value_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _actor_features(self, x):
        h = F.silu(self.actor_norm1(self.actor_fc1(x)))
        h = F.silu(self.actor_norm2(self.actor_fc2(h)))
        return h

    def _get_action_std(self, h):
        """State-dependent action std via RMSNorm→tanh SDE + learned log_std_param."""
        sde_raw = self.sde_fc(h)  # (batch, hidden_dim)
        # RMSNorm → linear → tanh: extra linear gives learned pre-tanh transform
        sde_latent = (self.sde_fc2(self.sde_norm(sde_raw)) / SDE_PRESCALE).tanh()  # (batch, hidden_dim)

        # Learned per-element log-std with offset initialization and clamping
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std_sq = log_std.exp().pow(2)  # (hidden_dim, act_dim)

        # SDE variance: sum over hidden dim of (latent^2 * std^2)
        action_var = (sde_latent.pow(2)) @ std_sq  # (batch, act_dim)
        action_std = (action_var + SDE_EPS).sqrt()
        return action_std

    def _critic_features(self, x):
        h = F.silu(self.critic_norm1(self.critic_fc1(x)))
        h = F.silu(self.critic_norm2(self.critic_fc2(h)))
        return h

    def get_value(self, x):
        h = self._critic_features(x)
        return self.value_out(h)

    def get_action_and_value(self, x, action=None):
        # Actor
        h = self._actor_features(x)
        action_mean = self.actor_out(h) * self.mean_scale
        action_std = self._get_action_std(h)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)


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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # SAC-style dual variables for adaptive SPO ε (separate for high/low advantages)
    # eps = exp(log_eps) ensures positive. Init from clip_coef values.
    log_eps_high = torch.tensor(np.log(args.clip_coef_high), dtype=torch.float32, device=device, requires_grad=True)
    log_eps_low = torch.tensor(np.log(args.clip_coef_low), dtype=torch.float32, device=device, requires_grad=True)
    eps_optimizer = optim.Adam([log_eps_high, log_eps_low], lr=args.spo_eps_lr)

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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
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
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        spo_penalty_sum = 0.0
        adv_dev_high_sum = 0.0
        adv_dev_low_sum = 0.0
        n_high = 0
        n_low = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio < (1 - log_eps_low.exp().item())) | (ratio > (1 + log_eps_high.exp().item())))
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Adaptive asymmetric SPO:
                # eps_high / eps_low are learned via dual descent on advantage-weighted targets
                eps_high = log_eps_high.exp().detach()
                eps_low = log_eps_low.exp().detach()
                eps = torch.where(mb_advantages >= 0, eps_high, eps_low)

                spo_surrogate = ratio * mb_advantages
                spo_penalty = torch.abs(mb_advantages) / (2 * eps) * (ratio - 1).pow(2)
                pg_loss = -(spo_surrogate - spo_penalty).mean()

                # Dual variable update: closed-loop advantage-weighted constraint
                # dev = E[|A|*(r-1)^2] — closed loop + advantage-weighted, bidirectional
                high_mask = mb_advantages >= 0
                low_mask = ~high_mask
                adv_ratio_sq = (torch.abs(mb_advantages) * (ratio - 1).pow(2)).detach()
                with torch.no_grad():
                    spo_penalty_sum += spo_penalty.mean().item()
                    if high_mask.any():
                        adv_dev_high_sum += adv_ratio_sq[high_mask].mean().item()
                        n_high += 1
                    if low_mask.any():
                        adv_dev_low_sum += adv_ratio_sq[low_mask].mean().item()
                        n_low += 1

                dev_h = adv_ratio_sq[high_mask].mean().detach() if high_mask.any() else torch.tensor(0.0, device=device)
                dev_l = adv_ratio_sq[low_mask].mean().detach() if low_mask.any() else torch.tensor(0.0, device=device)
                eps_loss_high = log_eps_high.exp() * (dev_h - args.spo_target_dev)
                eps_loss_low = log_eps_low.exp() * (dev_l - args.spo_target_dev)
                eps_loss = eps_loss_high + eps_loss_low

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef_low,
                        args.clip_coef_high,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Update policy/value
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Update eps dual variables (separate optimizer)
                eps_optimizer.zero_grad()
                eps_loss.backward()
                eps_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        n_updates = args.update_epochs * (args.batch_size // args.minibatch_size)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # SPO adaptive eps metrics
        writer.add_scalar("spo/eps_high", log_eps_high.exp().item(), global_step)
        writer.add_scalar("spo/eps_low", log_eps_low.exp().item(), global_step)
        writer.add_scalar("spo/penalty_mean", spo_penalty_sum / max(n_updates, 1), global_step)
        writer.add_scalar("spo/adv_dev_high", adv_dev_high_sum / max(n_high, 1), global_step)
        writer.add_scalar("spo/adv_dev_low", adv_dev_low_sum / max(n_low, 1), global_step)
        writer.add_scalar("spo/target_dev", args.spo_target_dev, global_step)
        # lstd-specific metrics
        log_std_eff = (agent.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        writer.add_scalar("tbot/log_std_mean", log_std_eff.mean().item(), global_step)
        writer.add_scalar("tbot/log_std_std", log_std_eff.std().item(), global_step)
        writer.add_scalar("tbot/mean_scale", agent.mean_scale.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
