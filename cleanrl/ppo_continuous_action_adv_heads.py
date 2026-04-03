# PPO with Advantage Decomposition Heads (V3: Directional Policy Shaping)
#
# Key idea: Add two auxiliary heads that provide directional gradient signal
# to the policy, decomposing the advantage into mean-direction and
# exploration-magnitude components.
#
# Architecture (shared encoder → 4 heads):
#   1. V(s)          — standard critic for GAE (unchanged PPO)
#   2. A_μ(s)        — "mean advantage" head: predicts which direction the
#                      mean should move, trained via advantage-weighted
#                      action residuals
#   3. A_σ(s)        — "exploration advantage" head: predicts per-dimension
#                      whether to explore more or less, trained via
#                      advantage-weighted variance signal
#   4. Actor          — standard policy (mean + logstd)
#
# The auxiliary heads provide soft gradient signal to the shared encoder
# and an optional auxiliary policy loss that pulls the mean toward the
# predicted improvement direction.
#
# Training targets:
#   A_μ target:  Â · (a - μ) / σ²  (the empirical policy gradient direction)
#   A_σ target:  Â · ((a - μ)² / σ² - 1)  (score function for variance)
#
# The actor receives an auxiliary loss:
#   L_aux = -λ_μ · cos_sim(∂μ, A_μ(s)) - λ_σ · (A_σ(s) · log σ).mean()
# This gently steers the policy in the direction the heads predict.

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
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Advantage head specific
    adv_head_coef: float = 0.25
    """coefficient for advantage head prediction losses"""
    mean_shaping_coef: float = 0.05
    """coefficient for mean-direction shaping auxiliary loss on actor"""
    std_shaping_coef: float = 0.02
    """coefficient for exploration-direction shaping auxiliary loss on actor"""
    shaping_warmup_frac: float = 0.1
    """fraction of training before shaping losses reach full strength"""

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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        # Shared encoder for critic heads
        self.critic_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Head 1: V(s) — standard value function
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)

        # Head 2: A_μ(s) — mean advantage direction (per action dimension)
        self.mean_adv_head = layer_init(nn.Linear(64, act_dim), std=0.01)

        # Head 3: A_σ(s) — exploration advantage (per action dimension)
        self.std_adv_head = layer_init(nn.Linear(64, act_dim), std=0.01)

        # Actor (separate encoder, as in standard PPO)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        features = self.critic_encoder(x)
        return self.value_head(features)

    def get_action_and_value(self, x, action=None):
        # Actor forward
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        # Critic forward (shared encoder, multiple heads)
        features = self.critic_encoder(x)
        value = self.value_head(features)
        mean_adv = self.mean_adv_head(features)
        std_adv = self.std_adv_head(features)

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
            mean_adv,
            std_adv,
            action_mean,
            action_std,
        )


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

        # Shaping coefficient warmup
        progress = iteration / args.num_iterations
        shaping_scale = min(1.0, progress / max(args.shaping_warmup_frac, 1e-8))

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _, _, _, _ = agent.get_action_and_value(next_obs)
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    pred_mean_adv,
                    pred_std_adv,
                    action_mean,
                    action_std,
                ) = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (standard PPO)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (standard)
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

                # === Advantage head losses ===
                # Compute targets for the advantage heads
                mb_actions = b_actions[mb_inds]
                with torch.no_grad():
                    action_residual = mb_actions - action_mean.detach()
                    action_var = (action_std.detach()) ** 2 + 1e-8

                    # Target for mean advantage: Â · (a - μ) / σ²
                    # This is the direction the score function points for the mean
                    mean_adv_target = mb_advantages.unsqueeze(-1) * action_residual / action_var

                    # Target for std advantage: Â · ((a - μ)² / σ² - 1)
                    # This is the score function for the log-std
                    std_adv_target = mb_advantages.unsqueeze(-1) * (
                        action_residual ** 2 / action_var - 1.0
                    )

                # Advantage head prediction losses (regression)
                mean_adv_loss = F.mse_loss(pred_mean_adv, mean_adv_target)
                std_adv_loss = F.mse_loss(pred_std_adv, std_adv_target)
                adv_head_loss = args.adv_head_coef * (mean_adv_loss + std_adv_loss)

                # === Shaping losses on the actor ===
                # Mean shaping: encourage the mean to move in the predicted direction
                # Use cosine similarity to be scale-invariant
                mean_shaping_loss = torch.zeros(1, device=device)
                if args.mean_shaping_coef > 0 and shaping_scale > 0:
                    # Detach the head predictions — we don't want actor gradients
                    # flowing back through the critic encoder
                    pred_dir = pred_mean_adv.detach()
                    # Dot product: ∂loss/∂action_mean = -pred_dir, pushing mean
                    # in the direction the advantage head predicts is beneficial.
                    mean_shaping_loss = -shaping_scale * args.mean_shaping_coef * (
                        pred_dir * action_mean
                    ).sum(-1).mean()

                # Std shaping: encourage std to adjust based on exploration advantage
                std_shaping_loss = torch.zeros(1, device=device)
                if args.std_shaping_coef > 0 and shaping_scale > 0:
                    pred_std_dir = pred_std_adv.detach()
                    # If pred_std_dir > 0 for a dimension, we want more exploration (larger std)
                    # If pred_std_dir < 0, we want less exploration (smaller std)
                    # Loss = -coef * pred_std_dir * logstd
                    # ∂loss/∂logstd = -coef * pred_std_dir
                    # Optimizer: logstd -= lr * (-coef * pred_std_dir) = logstd += lr * coef * pred_std_dir
                    # So positive pred_std_dir → increase logstd ✓
                    action_logstd = agent.actor_logstd.expand_as(action_mean)
                    std_shaping_loss = -shaping_scale * args.std_shaping_coef * (
                        pred_std_dir * action_logstd
                    ).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + adv_head_loss
                    + mean_shaping_loss
                    + std_shaping_loss
                )

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
        writer.add_scalar("losses/mean_adv_head_loss", mean_adv_loss.item(), global_step)
        writer.add_scalar("losses/std_adv_head_loss", std_adv_loss.item(), global_step)
        writer.add_scalar("losses/mean_shaping_loss", mean_shaping_loss.item(), global_step)
        writer.add_scalar("losses/std_shaping_loss", std_shaping_loss.item(), global_step)
        writer.add_scalar("charts/shaping_scale", shaping_scale, global_step)
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
