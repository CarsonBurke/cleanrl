# GAE-Pure Advantage Critic + PMPO-D4 PPO v1
#
# Combined hypothesis:
# - From advcritic_gaepure_v6: V(s) is only a TD(lambda)/GAE bootstrap; an
#   action-conditional A(s, a) is fit to those GAE targets, and only the
#   detached predicted A(s, a) is used as the policy-facing advantage.
# - From pmpo_d4_v1: replace ratio-clipped PPO with PMPO-D4 actor — squashed
#   actions via tanh + log|det J|, sign-split tanh(|adv|)-weighted per-action
#   log-prob updates, plus reverse KL(old || new).
# - Result: the actor sees a smooth, action-conditioned advantage (no raw GAE
#   leakage) and updates std under the gentler PMPO objective.
import os
import random
import time
from dataclasses import dataclass
from math import log

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.kl import kl_divergence
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
    """Toggle learning rate annealing for policy and advantage critic networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization (PMPO actor ignores this)."""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient (used for value clipping under PMPO)."""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the bootstrap value function."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the bootstrap value loss"""
    adv_coef: float = 0.5
    """coefficient of the advantage critic loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold (unused in PMPO branch)"""
    advantage_pretrain_epochs: int = 2
    """critic-only epochs on the current rollout before freezing policy advantages"""
    normalize_advantage_targets: bool = False
    """normalize GAE targets before fitting the advantage critic"""
    advantage_zero_mean_coef: float = 0.05
    """coefficient for E_pi[A(s,a)]^2 regularization"""
    advantage_zero_mean_samples: int = 4
    """number of policy action samples per state for zero-mean advantage regularization"""

    # PMPO-D4 specific arguments
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.value_baseline = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.advantage_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=0.01),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.register_buffer(
            "action_center",
            torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def get_value(self, x):
        return self.value_baseline(x)

    def get_action_distribution(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def squash_action(self, latent_action):
        return self.action_center + self.action_scale * torch.tanh(latent_action)

    def unsquash_action(self, action):
        normalized_action = ((action - self.action_center) / self.action_scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.atanh(normalized_action)

    def squash_log_abs_det_jacobian(self, latent_action):
        return torch.log(self.action_scale) + 2.0 * (log(2.0) - latent_action - nn.functional.softplus(-2.0 * latent_action))

    def get_squashed_action_and_value(self, x, action=None):
        latent_dist = self.get_action_distribution(x)
        if action is None:
            latent_action = latent_dist.sample()
            action = self.squash_action(latent_action)
        else:
            latent_action = self.unsquash_action(action)

        log_det = self.squash_log_abs_det_jacobian(latent_action)
        log_prob_per_dim = latent_dist.log_prob(latent_action) - log_det
        entropy_per_dim = latent_dist.entropy()
        return (
            action,
            log_prob_per_dim.sum(1),
            entropy_per_dim.sum(1),
            self.get_value(x),
            latent_dist,
            log_prob_per_dim,
        )

    def get_advantage(self, x, action):
        return self.advantage_critic(torch.cat((x, action), dim=1))


def advantage_zero_mean_loss(agent, obs, sample_count):
    if sample_count <= 0:
        return torch.zeros((), device=obs.device)

    with torch.no_grad():
        latent_dist = agent.get_action_distribution(obs)
        sampled_latent = latent_dist.sample((sample_count,))
        sampled_actions = agent.squash_action(sampled_latent)

    repeated_obs = obs.unsqueeze(0).expand(sample_count, *obs.shape).reshape(-1, obs.shape[-1])
    flat_actions = sampled_actions.reshape(-1, sampled_actions.shape[-1])
    sampled_advantages = agent.get_advantage(repeated_obs, flat_actions).view(sample_count, obs.shape[0])
    return sampled_advantages.mean(dim=0).pow(2).mean()


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
    old_action_means = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    old_action_stds = torch.ones((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

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
                action, logprob, _, value, action_dist, _ = agent.get_squashed_action_and_value(next_obs)
                values[step] = value.flatten()
                old_action_means[step] = action_dist.mean
                old_action_stds[step] = action_dist.stddev
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

        # Bootstrap GAE targets with V(s), then distill them into A(s, a).
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
            advantage_targets = advantages.clone()
            if args.normalize_advantage_targets:
                advantage_targets = (advantage_targets - advantage_targets.mean()) / (advantage_targets.std() + 1e-8)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantage_targets = advantage_targets.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_old_action_means = old_action_means.reshape((-1,) + envs.single_action_space.shape)
        b_old_action_stds = old_action_stds.reshape((-1,) + envs.single_action_space.shape)

        # Fit A(s, a) to the current rollout before freezing policy advantages.
        b_inds = np.arange(args.batch_size)
        adv_loss = torch.zeros((), device=device)
        zero_mean_loss = torch.zeros((), device=device)
        v_loss = torch.zeros((), device=device)
        for pretrain_epoch in range(args.advantage_pretrain_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                newadv = agent.get_advantage(b_obs[mb_inds], b_actions[mb_inds]).view(-1)
                adv_loss = 0.5 * ((newadv - b_advantage_targets[mb_inds]) ** 2).mean()
                zero_mean_loss = advantage_zero_mean_loss(agent, b_obs[mb_inds], args.advantage_zero_mean_samples)
                loss = adv_loss + args.advantage_zero_mean_coef * zero_mean_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        with torch.no_grad():
            b_predicted_advantages = agent.get_advantage(b_obs, b_actions).view(-1)
            b_advantages = b_predicted_advantages

        # Optimizing the policy and advantage critic with PMPO-D4 actor objective.
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

                _, newlogprob, entropy, _, new_dist, log_prob_per_dim = agent.get_squashed_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # PMPO-D4 actor with predicted-A advantage signal.
                mb_advantages = b_advantages[mb_inds]
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

                old_dist = Normal(b_old_action_means[mb_inds], b_old_action_stds[mb_inds])
                reverse_kl = kl_divergence(old_dist, new_dist).sum(-1).mean()
                reverse_kl_sum = reverse_kl_sum + reverse_kl.detach() * len(mb_inds)
                reverse_kl_count += len(mb_inds)
                if args.pmpo_kl_coef > 0.0:
                    if args.pmpo_reverse_kl:
                        kl_loss = reverse_kl
                    else:
                        kl_loss = kl_divergence(new_dist, old_dist).sum(-1).mean()
                    pg_loss = pg_loss + args.pmpo_kl_coef * kl_loss

                # Advantage critic loss
                newadv = agent.get_advantage(b_obs[mb_inds], b_actions[mb_inds]).view(-1)
                adv_loss = 0.5 * ((newadv - b_advantage_targets[mb_inds]) ** 2).mean()
                zero_mean_loss = advantage_zero_mean_loss(agent, b_obs[mb_inds], args.advantage_zero_mean_samples)

                # Bootstrap value loss
                newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
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
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + adv_loss * args.adv_coef
                    + zero_mean_loss * args.advantage_zero_mean_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        with torch.no_grad():
            final_advantages = agent.get_advantage(b_obs, b_actions).view(-1)
        y_pred, y_true = final_advantages.cpu().numpy(), b_advantage_targets.cpu().numpy()
        var_y = np.var(y_true)
        advantage_explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        value_explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if reverse_kl_count > 0:
            reverse_kl = reverse_kl_sum / reverse_kl_count

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/advantage_loss", adv_loss.item(), global_step)
        writer.add_scalar("losses/advantage_zero_mean_loss", zero_mean_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/reverse_kl", reverse_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/value_explained_variance", value_explained_var, global_step)
        writer.add_scalar("losses/advantage_explained_variance", advantage_explained_var, global_step)
        writer.add_scalar("losses/policy_advantage_std", b_advantages.std().item(), global_step)
        writer.add_scalar("losses/target_advantage_std", b_advantage_targets.std().item(), global_step)
        writer.add_scalar("losses/predicted_advantage_std", b_predicted_advantages.std().item(), global_step)
        writer.add_scalar("losses/raw_return_mean", b_returns.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
