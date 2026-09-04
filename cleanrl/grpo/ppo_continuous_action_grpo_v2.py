# GRPO for continuous control (Markov single-step policy).
#
# Faithful port of DeepSeek GRPO (Shao et al., 2024) adapted to a stateless
# (s -> a) policy. Changes vs grpo_v1:
#   (1) Per-trajectory scalar advantage. Group = all complete episodes in
#       one rollout across envs. A_i = (R_i - mean_R) / std_R, broadcast to
#       every transition in episode i. Replaces per-step cross-env z-score.
#   (2) Incomplete trailing episodes are masked out of the loss; episode
#       return accumulators carry across rollouts, so head-of-rollout
#       episodes that started in the previous rollout get a correct return.
#   (3) KL-to-reference penalty: snapshot pi_ref <- pi_theta every
#       ref_update_interval iterations and add -beta * D_KL^{k3}[pi || pi_ref]
#       to the objective (k3 estimator: exp(dlog) - dlog - 1).
#   (4) Per-trajectory loss normalization: weight each transition by
#       1/|episode|, so long episodes don't dominate the gradient.
#
# Actor is unchanged from hlgauss_silu_dlogstd_v1 (shared Tanh trunk with
# direct mean/logstd heads). No critic, no value head.
import copy
import os
import random
import time
from dataclasses import dataclass

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
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # GRPO-specific arguments
    beta_kl: float = 0.04
    """weight on the KL-to-reference regularizer"""
    ref_update_interval: int = 5
    """snapshot the reference policy every N iterations (1 = match old policy)"""

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
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean_head = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_logstd_head = layer_init(nn.Linear(64, act_dim), std=0.01)

    def get_action(self, x, action=None):
        trunk = self.actor_trunk(x)
        action_mean = self.actor_mean_head(trunk)
        action_logstd = torch.clamp(self.actor_logstd_head(trunk), -5.0, 2.0)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, 0.99) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    ref_agent = copy.deepcopy(agent).to(device)
    for p in ref_agent.parameters():
        p.requires_grad_(False)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup (no values — critic-free)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Episode-tracking state (persists across rollouts)
    ep_return_acc = np.zeros(args.num_envs, dtype=np.float64)
    ep_length_acc = np.zeros(args.num_envs, dtype=np.int64)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Snapshot reference policy periodically
        if (iteration - 1) % args.ref_update_interval == 0:
            ref_agent.load_state_dict(agent.state_dict())

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Per-rollout episode tracking: map each buffer transition to its
        # episode (or mark as belonging to a still-in-progress trailing one).
        episode_records = []  # list of (env_idx, t_start_local, t_end_local, total_return)
        ep_start_local = np.zeros(args.num_envs, dtype=np.int64)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())
            step_done = np.logical_or(terminations, truncations)
            rewards_buf[step] = torch.tensor(reward_np).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(step_done.astype(np.float32)).to(device)

            ep_return_acc += reward_np.astype(np.float64)
            ep_length_acc += 1
            for i in range(args.num_envs):
                if step_done[i]:
                    episode_records.append(
                        (i, int(ep_start_local[i]), step, float(ep_return_acc[i]), int(ep_length_acc[i]))
                    )
                    ep_return_acc[i] = 0.0
                    ep_length_acc[i] = 0
                    ep_start_local[i] = step + 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Build per-transition advantage, weight, and validity mask.
        #   advantage: scalar per episode, broadcast to all its transitions
        #   weight:    1/|episode| (for per-trajectory loss normalization)
        #   valid:     True if the transition belongs to a *complete* episode
        advantages = torch.zeros(args.num_steps, args.num_envs, device=device)
        weights = torch.zeros(args.num_steps, args.num_envs, device=device)
        valid_mask = torch.zeros(args.num_steps, args.num_envs, dtype=torch.bool, device=device)

        if len(episode_records) >= 2:
            ep_returns_np = np.array([rec[3] for rec in episode_records], dtype=np.float64)
            R_mean = ep_returns_np.mean()
            R_std = ep_returns_np.std() + 1e-8
            for (env_i, t_s, t_e, R, ep_len) in episode_records:
                adv = float((R - R_mean) / R_std)
                advantages[t_s : t_e + 1, env_i] = adv
                weights[t_s : t_e + 1, env_i] = 1.0 / float(ep_len)
                valid_mask[t_s : t_e + 1, env_i] = True
            writer.add_scalar("charts/group_return_mean", float(R_mean), global_step)
            writer.add_scalar("charts/group_return_std", float(R_std), global_step)
            writer.add_scalar("charts/group_size", len(episode_records), global_step)

        # Flatten and filter out invalid (in-progress) transitions
        b_obs_all = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs_all = logprobs.reshape(-1)
        b_actions_all = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages_all = advantages.reshape(-1)
        b_weights_all = weights.reshape(-1)
        b_valid_all = valid_mask.reshape(-1)

        valid_idx = torch.nonzero(b_valid_all, as_tuple=False).squeeze(-1)
        n_valid = valid_idx.numel()
        writer.add_scalar("charts/valid_fraction", n_valid / b_valid_all.numel(), global_step)

        if n_valid == 0:
            continue  # no complete episodes this rollout — skip update

        b_obs = b_obs_all[valid_idx]
        b_logprobs = b_logprobs_all[valid_idx]
        b_actions = b_actions_all[valid_idx]
        b_advantages = b_advantages_all[valid_idx]
        b_weights = b_weights_all[valid_idx]

        # Optimizing the policy network
        mb_size = max(1, n_valid // args.num_minibatches)
        b_inds = np.arange(n_valid)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, n_valid, mb_size):
                end = start + mb_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_adv = b_advantages[mb_inds]
                mb_w = b_weights[mb_inds]
                w_sum = mb_w.sum() + 1e-8

                # PPO-clipped surrogate, weighted per-trajectory
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_per = torch.max(pg_loss1, pg_loss2)
                pg_loss = (pg_per * mb_w).sum() / w_sum

                # KL to reference policy (k3 unbiased estimator, non-negative)
                with torch.no_grad():
                    _, ref_logprob, _ = ref_agent.get_action(b_obs[mb_inds], b_actions[mb_inds])
                log_ratio_ref = ref_logprob - newlogprob  # log(pi_ref / pi_theta)
                kl_per = log_ratio_ref.exp() - log_ratio_ref - 1.0
                kl_loss = (kl_per * mb_w).sum() / w_sum

                entropy_loss = (entropy * mb_w).sum() / w_sum

                loss = pg_loss + args.beta_kl * kl_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/kl_ref", kl_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
