# PMPO-D4 + ReluSq + categorical action bins v2.
#
# Hypothesis: Beta and TanhNormal PMPO can reward latent/density geometry
# without producing better actions. A categorical actor over fixed action bins
# makes the optimized object probability mass, not continuous density. Each
# action dimension has 51 bins over the env action range by default; sampled
# bin centers are sent to the environment as continuous scalars, while rollout
# storage keeps the integer bin ids for exact log-prob and KL evaluation.
#
# v2 replaces torch.distributions.kl_divergence(Categorical, Categorical)
# with a log-softmax categorical KL. PyTorch marks KL as inf when q.probs
# underflows to exactly zero; with finite logits the mathematical model still
# has finite log q. We do not clamp or floor probabilities.
#
# No delight gate. The PMPO objective is the direct D4-style sign-split
# categorical log-prob objective plus categorical reverse KL.

import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
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
    action_bins: int = 51
    """number of categorical bins per action dimension"""

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


def categorical_kl_from_logits(p_logits, q_logits):
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_probs = p_log_probs.exp()
    return (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)


class ReluSq(nn.Module):
    """f(x) = relu(x)^2 — x^2 for x>=0, 0 for x<0."""
    def forward(self, x):
        return torch.relu(x).square()


class Agent(nn.Module):
    def __init__(self, envs, action_bins=51):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.action_bins_count = action_bins

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor outputs independent categorical logits for each action dimension.
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, action_dim * action_bins), std=0.01),
        )
        action_low = torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        unit_bins = torch.linspace(0.0, 1.0, action_bins, dtype=torch.float32)
        action_values = action_low.unsqueeze(-1) + (action_high - action_low).unsqueeze(-1) * unit_bins
        self.register_buffer("action_values", action_values)

    def get_value(self, x):
        return self.critic(x)

    def get_action_distribution(self, x):
        logits = self.actor(x).view(-1, self.action_dim, self.action_bins_count)
        return Categorical(logits=logits)

    def _bins_to_action(self, action_bin_indices):
        values = self.action_values.unsqueeze(0).expand(action_bin_indices.shape[0], -1, -1)
        return values.gather(-1, action_bin_indices.unsqueeze(-1)).squeeze(-1)

    def get_action_and_value(self, x, action_bin_indices=None):
        # PPO branch — same categorical machinery; PPO loss uses clipped ratio.
        dist = self.get_action_distribution(x)
        if action_bin_indices is None:
            action_bin_indices = dist.sample()
        action = self._bins_to_action(action_bin_indices)
        log_prob_per_dim = dist.log_prob(action_bin_indices)
        entropy_per_dim = dist.entropy()
        return action, action_bin_indices, log_prob_per_dim.sum(1), entropy_per_dim.sum(1), self.critic(x)

    def get_squashed_action_and_value(self, x, action_bin_indices=None):
        # PMPO branch — returns the categorical dist for KL and per-dim log_probs for masking.
        dist = self.get_action_distribution(x)
        if action_bin_indices is None:
            action_bin_indices = dist.sample()
        action = self._bins_to_action(action_bin_indices)
        log_prob_per_dim = dist.log_prob(action_bin_indices)
        entropy_per_dim = dist.entropy()
        return (
            action,
            action_bin_indices,
            log_prob_per_dim.sum(1),
            entropy_per_dim.sum(1),
            self.critic(x),
            dist,
            log_prob_per_dim,
        )


def evaluate_policy(model_path, make_env, env_id, eval_episodes, run_name, model, device, gamma, policy_objective, action_bins):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, True, run_name, gamma)])
    agent = model(envs, action_bins).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            if policy_objective == "pmpo":
                actions, _, _, _, _, _, _ = agent.get_squashed_action_and_value(obs_tensor)
            else:
                actions, _, _, _, _ = agent.get_action_and_value(obs_tensor)
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

    agent = Agent(envs, args.action_bins).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_bin_indices = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    old_action_logits = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape + (args.action_bins,)
    ).to(device)

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
                    action, action_bins_idx, logprob, _, value, action_dist, _ = agent.get_squashed_action_and_value(next_obs)
                else:
                    action, action_bins_idx, logprob, _, value = agent.get_action_and_value(next_obs)
                    action_dist = agent.get_action_distribution(next_obs)
                values[step] = value.flatten()
                old_action_logits[step] = action_dist.logits
            actions[step] = action
            action_bin_indices[step] = action_bins_idx
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
        b_action_bin_indices = action_bin_indices.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_old_action_logits = old_action_logits.reshape((-1,) + envs.single_action_space.shape + (args.action_bins,))

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
                    _, _, newlogprob, entropy, newvalue, new_dist, log_prob_per_dim = agent.get_squashed_action_and_value(
                        b_obs[mb_inds], b_action_bin_indices[mb_inds]
                    )
                else:
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_action_bin_indices[mb_inds]
                    )
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
                    adv_per_dim = mb_advantages.unsqueeze(-1)  # [N, 1] broadcasts across action dims
                    adv_weight = adv_per_dim.tanh().abs()

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

                    old_logits = b_old_action_logits[mb_inds]
                    reverse_kl = categorical_kl_from_logits(old_logits, new_dist.logits).sum(-1).mean()
                    reverse_kl_sum = reverse_kl_sum + reverse_kl.detach() * len(mb_inds)
                    reverse_kl_count += len(mb_inds)
                    if args.pmpo_kl_coef > 0.0:
                        if args.pmpo_reverse_kl:
                            kl_loss = reverse_kl
                        else:
                            kl_loss = categorical_kl_from_logits(new_dist.logits, old_logits).sum(-1).mean()
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

        # Categorical actor diagnostics
        with torch.no_grad():
            old_probs = b_old_action_logits.softmax(dim=-1)
            old_log_probs = F.log_softmax(b_old_action_logits, dim=-1)
            mean_max_prob = old_probs.max(dim=-1).values.mean().item()
            mean_bin_entropy = Categorical(logits=b_old_action_logits).entropy().mean().item()
            mean_selected_bin = b_action_bin_indices.float().mean().item()
            mean_logit_range = (b_old_action_logits.max(dim=-1).values - b_old_action_logits.min(dim=-1).values).mean().item()
            mean_min_log_prob = old_log_probs.min(dim=-1).values.mean().item()
            mean_edge_prob = (old_probs[..., 0] + old_probs[..., -1]).mean().item()
            selected_edge_frac = (
                ((b_action_bin_indices == 0) | (b_action_bin_indices == args.action_bins - 1)).float().mean().item()
            )

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
        writer.add_scalar("diag/cat_mean_max_prob", mean_max_prob, global_step)
        writer.add_scalar("diag/cat_mean_bin_entropy", mean_bin_entropy, global_step)
        writer.add_scalar("diag/cat_mean_selected_bin", mean_selected_bin, global_step)
        writer.add_scalar("diag/cat_mean_logit_range", mean_logit_range, global_step)
        writer.add_scalar("diag/cat_mean_min_log_prob", mean_min_log_prob, global_step)
        writer.add_scalar("diag/cat_mean_edge_prob", mean_edge_prob, global_step)
        writer.add_scalar("diag/cat_selected_edge_frac", selected_edge_frac, global_step)
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
            action_bins=args.action_bins,
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
