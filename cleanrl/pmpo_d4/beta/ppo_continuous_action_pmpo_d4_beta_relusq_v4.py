# PMPO-D4 + ReluSq + Beta v4 — state-independent concentration (PPO-style split).
#
# v1/v2/v3 had α and β both as state-dependent network outputs, so concentration
# (α+β) was also state-dependent. On Hopper this drove concentration runaway:
# early bad states → PMPO neg_loss suppresses log_prob across the action support
# → policy concentrates per-state on locally-best action ("do nothing") → α+β
# grows → exploration dies → reward stuck at 13. PPO+Gaussian doesn't suffer
# this because actor_logstd is a state-INDEPENDENT free parameter — global
# noise scale that changes only via aggregated gradient, slowly. Della Libera's
# Beta recipe inherits SAC's tolerance via fixed entropy bonus α=0.2; PMPO
# doesn't, so we need the structural fix.
#
# v4 mirrors PPO's location/scale split for Beta:
#   μ = sigmoid(head_μ)                      # state-dependent location ∈ (0,1)
#   c = exp(actor_logc)                      # state-INDEPENDENT concentration
#   α = 1 + c·μ
#   β = 1 + c·(1 - μ)
# Properties:
#   - α, β ≥ 1 by construction (no clamps, no caps, no soft-clips)
#   - α + β = 2 + c, controlled only by the global parameter logc
#   - Beta mean = (1+cμ)/(2+c) ≈ μ for moderate-to-large c, exact at μ=0.5
#   - Concentration changes globally and slowly — analog of σ in PPO+Gaussian
#   - No dying-gradient pathology (no soft-clip in the parametrization)
#
# Init: actor_logc=0 → c=1; head_μ≈0 → μ≈0.5. Beta(1.5, 1.5) at μ=0.5, broad,
# centered. action_low + (action_high - action_low) * μ = action center.
#
# Sample clamp to [1e-7, 1-1e-7] for log/digamma stability.
# Linear remap [0,1] → [low, high] (constant Jacobian, cancels in ratios).
# Reverse-KL between Betas via torch.distributions.kl_divergence (closed form).

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
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter


SAMPLE_EPS = 1e-7


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
    total_timesteps: int = 1000000
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
    """Toggles PPO advantage normalization. PMPO intentionally ignores this."""
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
    target_kl: float = None
    """PPO-only target ratio KL threshold"""

    # PMPO-D4 specific arguments
    policy_objective: Literal["pmpo", "ppo"] = "pmpo"
    """actor objective: pmpo is D4-style PMPO, ppo is base clipped PPO"""
    pmpo_pos_to_neg_weight: float = 0.5
    """PMPO positive-advantage weight; negative side uses 1 - this value"""
    pmpo_kl_coef: float = 0.3
    """reverse KL(old rollout policy || new policy) coefficient"""
    pmpo_reverse_kl: bool = True
    """use KL(old||new); false uses KL(new||old) for ablations"""

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
    """f(x) = relu(x)^2 — x^2 for x>=0, 0 for x<0."""
    def forward(self, x):
        return torch.relu(x).square()


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # State-dependent location head (μ ∈ (0,1) via sigmoid).
        self.actor_mu = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # State-INDEPENDENT log concentration (analog of actor_logstd in PPO+Gaussian).
        self.actor_logc = nn.Parameter(torch.zeros(1, action_dim))
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_distribution(self, x):
        mu = torch.sigmoid(self.actor_mu(x))
        c = torch.exp(self.actor_logc).expand_as(mu)
        alpha = 1.0 + c * mu
        beta = 1.0 + c * (1.0 - mu)
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        return ((action - self.action_low) / (self.action_high - self.action_low)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, action=None):
        # PPO branch — same Beta machinery; PPO loss uses clipped ratio (no Jacobian needed).
        dist = self.get_action_distribution(x)
        if action is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z)
        else:
            z = self._action_to_z(action)
        log_prob_per_dim = dist.log_prob(z)
        entropy_per_dim = dist.entropy()
        return action, log_prob_per_dim.sum(1), entropy_per_dim.sum(1), self.critic(x)

    def get_squashed_action_and_value(self, x, action=None):
        # PMPO branch — returns the latent (Beta) dist for KL and per-dim log_probs for masking.
        dist = self.get_action_distribution(x)
        if action is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z)
        else:
            z = self._action_to_z(action)
        log_prob_per_dim = dist.log_prob(z)
        entropy_per_dim = dist.entropy()
        return (
            action,
            log_prob_per_dim.sum(1),
            entropy_per_dim.sum(1),
            self.critic(x),
            dist,
            log_prob_per_dim,
        )


def evaluate_policy(model_path, make_env, env_id, eval_episodes, run_name, model, device, gamma, policy_objective):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, True, run_name, gamma)])
    agent = model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            if policy_objective == "pmpo":
                actions, _, _, _, _, _ = agent.get_squashed_action_and_value(obs_tensor)
            else:
                actions, _, _, _ = agent.get_action_and_value(obs_tensor)
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    envs.close()
    return episodic_returns


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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Beta-policy stores α, β (concentration1, concentration0) per dim.
    old_alphas = torch.ones((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    old_betas = torch.ones((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

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
                if args.policy_objective == "pmpo":
                    action, logprob, _, value, action_dist, _ = agent.get_squashed_action_and_value(next_obs)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    action_dist = agent.get_action_distribution(next_obs)
                values[step] = value.flatten()
                old_alphas[step] = action_dist.concentration1
                old_betas[step] = action_dist.concentration0
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
        b_old_alphas = old_alphas.reshape((-1,) + envs.single_action_space.shape)
        b_old_betas = old_betas.reshape((-1,) + envs.single_action_space.shape)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        reverse_kl = torch.zeros((), device=device)
        reverse_kl_sum = torch.zeros((), device=device)
        reverse_kl_count = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.policy_objective == "pmpo":
                    _, newlogprob, entropy, newvalue, new_dist, log_prob_per_dim = agent.get_squashed_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    new_dist = None
                    log_prob_per_dim = None
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.policy_objective == "ppo":
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:
                    adv_weight = mb_advantages.tanh().abs().unsqueeze(-1)
                    signed_log_probs = log_prob_per_dim * adv_weight
                    pos_mask = (mb_advantages >= 0).unsqueeze(-1).expand_as(log_prob_per_dim)
                    neg_mask = (mb_advantages < 0).unsqueeze(-1).expand_as(log_prob_per_dim)

                    pos_count = pos_mask.float().sum().clamp_min(1.0)
                    neg_count = neg_mask.float().sum().clamp_min(1.0)
                    pos_loss = signed_log_probs[pos_mask].sum() / pos_count
                    neg_loss = signed_log_probs[neg_mask].sum() / neg_count
                    pg_loss = (
                        -args.pmpo_pos_to_neg_weight * pos_loss
                        + (1.0 - args.pmpo_pos_to_neg_weight) * neg_loss
                    )

                    old_dist = Beta(b_old_alphas[mb_inds], b_old_betas[mb_inds])
                    reverse_kl = kl_divergence(old_dist, new_dist).sum(-1).mean()
                    reverse_kl_sum = reverse_kl_sum + reverse_kl.detach() * len(mb_inds)
                    reverse_kl_count += len(mb_inds)
                    if args.pmpo_kl_coef > 0.0:
                        if args.pmpo_reverse_kl:
                            kl_loss = reverse_kl
                        else:
                            kl_loss = kl_divergence(new_dist, old_dist).sum(-1).mean()
                        pg_loss = pg_loss + args.pmpo_kl_coef * kl_loss

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

            if args.policy_objective == "ppo" and args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.policy_objective == "pmpo" and reverse_kl_count > 0:
            reverse_kl = reverse_kl_sum / reverse_kl_count
        policy_kl = approx_kl if args.policy_objective == "ppo" else reverse_kl

        # Beta diagnostics
        with torch.no_grad():
            mean_alpha = b_old_alphas.mean().item()
            mean_beta = b_old_betas.mean().item()
            sum_conc = (b_old_alphas + b_old_betas).mean().item()
            logc_mean = agent.actor_logc.mean().item()
            c_mean = float(np.exp(logc_mean))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/reverse_kl", reverse_kl.item(), global_step)
        writer.add_scalar("losses/policy_kl", policy_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/beta_mean_alpha", mean_alpha, global_step)
        writer.add_scalar("diag/beta_mean_beta", mean_beta, global_step)
        writer.add_scalar("diag/beta_concentration_sum", sum_conc, global_step)
        writer.add_scalar("diag/actor_logc_mean", logc_mean, global_step)
        writer.add_scalar("diag/actor_c_mean", c_mean, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        episodic_returns = evaluate_policy(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            model=Agent,
            device=device,
            gamma=args.gamma,
            policy_objective=args.policy_objective,
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
