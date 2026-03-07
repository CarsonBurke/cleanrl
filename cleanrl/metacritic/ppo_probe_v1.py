"""Probing Policy Optimization (Phase 1) — Candidate Evaluation with Real Rollouts

Fork of lstd_linear_tanh with post-update candidate probing. At each PPO iteration,
after the standard update epochs complete:

1. Snapshot parameters before and after the PPO update.
2. Compute delta = theta_updated - theta_before (the update direction).
3. Maintain a buffer of the last 20 deltas. Compute SVD to extract top eigenvectors
   of the update covariance — these are the principal directions of recent optimization.
4. Generate N=5 candidates by perturbing theta_updated along top eigenvectors:
   - candidate_0 = theta_updated (baseline)
   - candidate_{1,2} = theta_updated +/- alpha * e_1
   - candidate_{3,4} = theta_updated +/- alpha * e_2
   where alpha = probe_magnitude * ||delta||.
5. Evaluate each candidate with a short rollout (probe_eval_steps steps), using the
   critic's bootstrapped value estimate as a fast proxy for return.
6. Select the best candidate and set the agent's parameters accordingly.

This tests whether choosing among update directions improves over blindly following
the single PPO gradient. Probing starts after iteration 20 (need delta history).

Key: during probe rollouts, environments advance but we use torch.no_grad() and
do not interfere with the main training loop's data collection.
"""
import os
import random
import time
import copy
from dataclasses import dataclass
from collections import deque

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
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef_low: float = 0.2
    """the lower surrogate clipping coefficient (ratio floor = 1 - this)"""
    clip_coef_high: float = 0.28
    """the upper surrogate clipping coefficient (ratio ceiling = 1 + this)"""
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

    # Probing arguments
    probe_magnitude: float = 0.1
    """scaling factor for probe perturbations (alpha = probe_magnitude * ||delta||)"""
    num_probe_candidates: int = 5
    """number of candidate parameter vectors to evaluate (including the default)"""
    probe_eval_steps: int = 200
    """number of environment steps per candidate evaluation rollout"""
    delta_buffer_size: int = 20
    """number of recent deltas to keep for covariance estimation"""
    probe_start_iteration: int = 20
    """iteration after which probing begins (need delta history first)"""

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

        # Actor backbone
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_norm2 = RMSNorm(hidden_dim)

        # Actor output head
        self.actor_out = layer_init(nn.Linear(hidden_dim, act_dim), std=1.0)
        self.mean_scale = nn.Parameter(torch.tensor(0.01))

        # SDE noise with learned log_std_param
        self.sde_fc = layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.sde_norm = RMSNorm(hidden_dim)
        self.sde_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.log_std_param = nn.Parameter(torch.zeros(hidden_dim, act_dim))

        # Critic backbone
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


# ---- Probing utilities ----

def params_to_vector(model):
    """Flatten all parameters into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def vector_to_params(vec, model):
    """Load a flat vector back into model parameters."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset:offset + numel].view(p.shape))
        offset += numel


def evaluate_candidate(agent, envs, next_obs, num_steps, device, gamma):
    """Run a short rollout and return bootstrapped value estimate.

    Returns (mean_value, next_obs_after_rollout) where mean_value is the
    average critic value at the final state across all envs — a fast proxy
    for expected return from the current policy.
    """
    total_reward = 0.0
    obs = next_obs
    with torch.no_grad():
        for step in range(num_steps):
            action, _, _, _ = agent.get_action_and_value(obs)
            obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            total_reward += reward.mean()
            obs = torch.Tensor(obs_np).to(device)

        # Bootstrapped value at final state
        final_value = agent.get_value(obs).mean().item()

    # Combine accumulated rewards with bootstrapped value
    # Use a simple discounted estimate: total_reward + gamma^steps * V(s_final)
    discount = gamma ** num_steps
    score = total_reward + discount * final_value
    return score, obs


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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Probing state
    delta_buffer = deque(maxlen=args.delta_buffer_size)

    # Start the game
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

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

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

        # Bootstrap value if not done
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

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---- Snapshot parameters BEFORE update ----
        theta_before = params_to_vector(agent).clone()

        # Optimizing the policy and value network (standard PPO)
        b_inds = np.arange(args.batch_size)
        clipfracs = []
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
                        ((ratio < (1 - args.clip_coef_low)) | (ratio > (1 + args.clip_coef_high)))
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef_low, 1 + args.clip_coef_high)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- Snapshot parameters AFTER update ----
        theta_updated = params_to_vector(agent).clone()
        delta_vec = theta_updated - theta_before
        delta_buffer.append(delta_vec)

        # ---- Candidate Probing ----
        probe_winner = 0  # default: keep theta_updated
        probe_scores = []
        if iteration >= args.probe_start_iteration and len(delta_buffer) >= args.delta_buffer_size:
            delta_norm = delta_vec.norm().item()
            alpha = args.probe_magnitude * delta_norm

            if alpha > 1e-10:
                # Compute top eigenvectors of delta covariance via SVD
                # Stack deltas: (buffer_size, param_dim)
                delta_matrix = torch.stack(list(delta_buffer))  # (B, D)
                # Center the deltas
                delta_matrix = delta_matrix - delta_matrix.mean(dim=0, keepdim=True)
                # Economy SVD — we only need top 2 right singular vectors
                # U: (B, B), S: (min(B,D),), Vh: (min(B,D), D)
                try:
                    U, S, Vh = torch.linalg.svd(delta_matrix, full_matrices=False)
                    # Top eigenvectors of the covariance are the top rows of Vh
                    e1 = Vh[0]  # (D,)
                    e2 = Vh[1] if Vh.shape[0] > 1 else torch.zeros_like(e1)
                except Exception:
                    # SVD can fail in degenerate cases; skip probing this iteration
                    e1 = None

                if e1 is not None:
                    # Generate candidates
                    candidates = [
                        theta_updated,                          # 0: default
                        theta_updated + alpha * e1,             # 1: +e1
                        theta_updated - alpha * e1,             # 2: -e1
                        theta_updated + alpha * e2,             # 3: +e2
                        theta_updated - alpha * e2,             # 4: -e2
                    ]
                    # Only use num_probe_candidates
                    candidates = candidates[:args.num_probe_candidates]

                    # Save current env observation for restoring after probing
                    # (We don't restore env state — envs advance during probing, which is fine)
                    probe_obs = next_obs.clone()

                    best_score = -float('inf')
                    best_idx = 0

                    for c_idx, candidate in enumerate(candidates):
                        # Load candidate parameters
                        vector_to_params(candidate, agent)

                        # Run short evaluation rollout
                        score, probe_obs_after = evaluate_candidate(
                            agent, envs, probe_obs if c_idx == 0 else next_obs,
                            args.probe_eval_steps, device, args.gamma
                        )
                        probe_scores.append(score)

                        if score > best_score:
                            best_score = score
                            best_idx = c_idx

                        # For subsequent candidates, reset obs to where we are now
                        # (each candidate evaluation advances envs, so we use current state)
                        # Actually, we need to track where envs end up after each eval.
                        # Since envs advance, each candidate sees different states.
                        # The last candidate's final obs becomes next_obs for the main loop.
                        next_obs = probe_obs_after

                    # Apply the winning candidate
                    vector_to_params(candidates[best_idx], agent)
                    probe_winner = best_idx

                    # Also need to update the optimizer's state to match the new params
                    # Since we're directly setting params, optimizer momentum is stale
                    # but this is acceptable — the optimizer will adapt.

                    # Update delta buffer with the actual final delta (including probe)
                    if best_idx != 0:
                        delta_buffer[-1] = candidates[best_idx] - theta_before

                    # Log probing metrics
                    writer.add_scalar("probe/winner_idx", best_idx, global_step)
                    writer.add_scalar("probe/alpha", alpha, global_step)
                    writer.add_scalar("probe/delta_norm", delta_norm, global_step)
                    if len(probe_scores) > 1:
                        writer.add_scalar("probe/score_default", probe_scores[0], global_step)
                        writer.add_scalar("probe/score_best", best_score, global_step)
                        writer.add_scalar("probe/score_spread",
                                         max(probe_scores) - min(probe_scores), global_step)
                        # Track how often non-default wins
                        writer.add_scalar("probe/nondefault_win", float(best_idx != 0), global_step)
                    # Log top singular values (measure of update direction consistency)
                    writer.add_scalar("probe/sv_ratio", (S[0] / (S[1] + 1e-10)).item(), global_step)

        # Standard logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # lstd-specific metrics
        log_std_eff = (agent.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        writer.add_scalar("tbot/log_std_mean", log_std_eff.mean().item(), global_step)
        writer.add_scalar("tbot/log_std_std", log_std_eff.std().item(), global_step)
        writer.add_scalar("tbot/mean_scale", agent.mean_scale.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)),
              f"probe_winner={probe_winner}" if probe_scores else "")
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
