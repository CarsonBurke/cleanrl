# PMPO + LSTD: Policy Mirror Primal Optimization with Learned State-Dependent exploration.
#
# Combines two ideas:
# 1. PMPO (https://arxiv.org/abs/2410.04166v1): sign-based advantage splitting with KL
#    regularization instead of PPO's clipped surrogate objective.
# 2. LSTD (lstd_linear_tanh): state-dependent exploration via RMSNorm+tanh SDE latent
#    with learned per-element log_std_param matrix and shared actor weight matrix.
#
# Architecture from LSTD:
# - RMSNorm + SiLU backbone, mean_scale decoupling
# - SDE noise: sde_fc -> sde_norm -> sde_fc2 -> /prescale -> tanh -> matmul with learned std
# - Learned log_std_param [hidden_dim, act_dim] with LOG_STD_INIT offset and clamping
#
# Policy optimization from PMPO:
# - Split advantages into positive/negative masks
# - Maximize log_prob for positive, minimize for negative (equal weight by default)
# - KL divergence regularization between old and new policy (using stored action_std)
# - No PPO clipping on policy loss (value clipping retained)

import copy
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
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter


LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
SDE_EPS = 1e-6
SDE_PRESCALE = 1.5


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
    norm_adv: bool = False
    """Toggles advantages normalization (off by default for PMPO)"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient (used for value clipping only)"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PMPO specific arguments
    pmpo_alpha: float = 0.5
    """weight for positive advantages vs negative advantages (0.5 = equal)"""
    pmpo_kl_coef: float = 0.3
    """coefficient for KL divergence regularization"""
    pmpo_reverse_kl: bool = True
    """use reverse KL (KL(old||new)) instead of forward KL (KL(new||old))"""

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
        """State-dependent action std via RMSNorm->tanh SDE + learned log_std_param."""
        sde_raw = self.sde_fc(h)
        sde_latent = (self.sde_fc2(self.sde_norm(sde_raw)) / SDE_PRESCALE).tanh()

        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std_sq = log_std.exp().pow(2)

        action_var = (sde_latent.pow(2)) @ std_sq
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
        h = self._actor_features(x)
        action_mean = self.actor_out(h) * self.mean_scale
        action_std = self._get_action_std(h)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)

    def get_action_value_and_dist(self, x, action):
        """Single forward pass returning per-dim log_probs, entropy, value, and dist for PMPO."""
        h = self._actor_features(x)
        action_mean = self.actor_out(h) * self.mean_scale
        action_std = self._get_action_std(h)

        dist = Normal(action_mean, action_std)
        log_prob_per_dim = dist.log_prob(action)
        entropy = dist.entropy().sum(1)
        return log_prob_per_dim, entropy, self.get_value(x), dist


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

    act_dim = np.prod(envs.single_action_space.shape)

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

        # Frozen prior: deepcopy actor once before all epochs (d4-style)
        # Must copy all actor components since std is state-dependent
        prior_agent = copy.deepcopy(agent).eval()
        for p in prior_agent.parameters():
            p.requires_grad = False

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Single forward pass: per-dim log_probs, entropy, value, and current dist
                log_prob_per_dim, entropy, newvalue, new_dist = agent.get_action_value_and_dist(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                newlogprob = log_prob_per_dim.sum(1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PMPO Policy loss (per-dim log_probs with masked mean)
                pos_mask = mb_advantages >= 0
                neg_mask = ~pos_mask

                pos_mask_expanded = pos_mask.unsqueeze(-1).expand_as(log_prob_per_dim)
                neg_mask_expanded = neg_mask.unsqueeze(-1).expand_as(log_prob_per_dim)

                alpha = args.pmpo_alpha

                pos_count = pos_mask_expanded.sum()
                neg_count = neg_mask_expanded.sum()

                pos_loss = log_prob_per_dim[pos_mask_expanded].sum() / pos_count.clamp(min=1)
                neg_loss = -log_prob_per_dim[neg_mask_expanded].sum() / neg_count.clamp(min=1)

                pg_loss = -(alpha * pos_loss + (1.0 - alpha) * neg_loss)

                # KL divergence regularization against frozen prior (d4-style)
                if args.pmpo_kl_coef > 0.0:
                    with torch.no_grad():
                        _, _, _, prior_dist = prior_agent.get_action_value_and_dist(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )

                    if args.pmpo_reverse_kl:
                        kl = kl_divergence(prior_dist, new_dist).sum(-1).mean()
                    else:
                        kl = kl_divergence(new_dist, prior_dist).sum(-1).mean()

                    pg_loss = pg_loss + args.pmpo_kl_coef * kl

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
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
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # LSTD-specific metrics
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
