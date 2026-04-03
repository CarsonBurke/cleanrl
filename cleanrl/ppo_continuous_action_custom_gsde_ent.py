# Latent-Space Noise + Shared W + Entropy Band (v3c)
#
# Moves noise from action space to latent space. Instead of learning a full
# Cholesky factor directly (v2), we learn per-latent-dim noise scales σ(s)
# and derive cross-actuator correlations from the normalized mean weight W_dir.
#
# Why: Action-space Cholesky has a signal problem — off-diagonal elements get
# competing gradients from entropy (grow |Σ|) and advantages (noisy). Entropy
# wins → off-diags blow up. Bounding (tanh) fixes stability but caps performance.
#
# Architecture:
#   Mean:  obs → trunk → latent → W @ latent + b → action_mean
#   Noise: obs → noise_trunk → noise_fc → log_σ → σ = exp(clamp(log_σ))
#   Corr:  W_dir = W / ||W|| (row-normalized mean weight, NOT detached)
#   Cov:   WS = W_dir * σ; Σ = WS @ WS^T + εI
#   Sample: action = mean + chol(Σ) @ z
#
# Key design:
#   - σ_j has independent per-dim gradient (no coupled off-diagonal mess)
#   - W_dir not detached → PG shapes correlation structure through W direction
#   - Cholesky used only for log_prob math, not as learned parameterization
#   - 64 independent σ params vs 21 coupled Cholesky params for 6 actions

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
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Entropy band (floor + ceiling)
    target_ent_per_dim: float = 0.6
    """target entropy per action dimension (ceiling)"""
    ent_floor_per_dim: float = 0.4
    """minimum entropy per action dimension (floor) — prevents exploration collapse"""
    ent_ceiling_coef: float = 0.5
    """coefficient for entropy ceiling penalty"""
    ent_floor_coef: float = 0.5
    """coefficient for entropy floor penalty"""

    # Latent noise
    sigma_clamp_max: float = 5.0
    """max value for log_sigma (exp(5)≈148, safety ceiling)"""

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
    def __init__(self, envs, sigma_clamp_max=5.0):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.action_dim = action_dim
        self.hidden_dim = 64
        self.sigma_clamp_max = sigma_clamp_max
        self._log2pi = np.log(2 * np.pi)

        # Identity buffer for jitter
        self.register_buffer("eye", torch.eye(action_dim))

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            nn.RMSNorm(64),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor mean pathway
        self.actor_mean_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)

        # Noise pathway — separate trunk for state-dependent noise scale
        self.noise_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            nn.RMSNorm(64),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            nn.RMSNorm(64),
        )
        # Outputs log_σ per latent dim (64 dims)
        self.noise_fc = layer_init(nn.Linear(64, 64), std=0.01)
        # Init bias: want initial effective action std ≈ 0.5
        # W_dir is row-normalized to unit norm, so effective std ≈ σ directly
        self.noise_fc.bias.data.fill_(np.log(0.5))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        B = x.shape[0]
        ad = self.action_dim

        # Mean pathway
        latent = self.actor_mean_trunk(x)
        action_mean = self.actor_mean(latent)

        # Per-latent-dim noise scale (state-dependent)
        log_sigma = self.noise_fc(self.noise_trunk(x))  # [B, 64]
        sigma = torch.exp(log_sigma.clamp(max=self.sigma_clamp_max))  # [B, 64]

        # Correlation structure from normalized mean weight
        W = self.actor_mean.weight  # [ad, 64]
        W_dir = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # [ad, 64] row-normalized

        # Gram covariance: Σ = (W_dir · diag(σ)) @ (W_dir · diag(σ))^T + εI
        WS = W_dir.unsqueeze(0) * sigma.unsqueeze(1)  # [B, ad, 64]
        gram = torch.bmm(WS, WS.transpose(1, 2))  # [B, ad, ad]
        cov = gram + 1e-4 * self.eye  # [B, ad, ad]
        chol = torch.linalg.cholesky(cov)  # [B, ad, ad]

        if action is None:
            z = torch.randn(B, ad, 1, device=x.device)
            action = action_mean + torch.bmm(chol, z).squeeze(-1)

        # Log prob via Cholesky
        diff = (action - action_mean).unsqueeze(-1)  # [B, ad, 1]
        solved = torch.linalg.solve_triangular(chol, diff, upper=False)
        mahal = solved.squeeze(-1).pow(2).sum(dim=-1)  # [B]
        log_diag = chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [B]
        log_prob = -0.5 * (ad * self._log2pi + 2 * log_diag + mahal)

        # Entropy = 0.5 * (d * (1 + log2π) + 2 * Σ log(L_ii))
        entropy = 0.5 * (ad * (1.0 + self._log2pi) + 2 * log_diag)

        return action, log_prob, entropy, self.critic(x)


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    action_dim = np.prod(envs.single_action_space.shape)
    target_entropy = args.target_ent_per_dim * action_dim
    floor_entropy = args.ent_floor_per_dim * action_dim

    agent = Agent(envs, sigma_clamp_max=args.sigma_clamp_max).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
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

        # Bootstrap
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

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
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
                    clipfracs += [((ratio < (1 - args.clip_coef_low)) | (ratio > (1 + args.clip_coef_high))).float().mean().item()]

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
                ent_excess = F.relu(entropy_loss - target_entropy)
                ent_ceiling_loss = ent_excess ** 2
                ent_deficit = F.relu(floor_entropy - entropy_loss)
                ent_floor_loss = ent_deficit ** 2

                loss = pg_loss + v_loss * args.vf_coef + args.ent_ceiling_coef * ent_ceiling_loss + args.ent_floor_coef * ent_floor_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

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
        writer.add_scalar("ent_ceiling/penalty", ent_ceiling_loss.item(), global_step)
        writer.add_scalar("ent_ceiling/target", target_entropy, global_step)
        writer.add_scalar("ent_floor/penalty", ent_floor_loss.item(), global_step)
        writer.add_scalar("ent_floor/target", floor_entropy, global_step)
        with torch.no_grad():
            # Latent noise diagnostics from last minibatch
            log_sigma = agent.noise_fc(agent.noise_trunk(b_obs[mb_inds]))
            sigma = torch.exp(log_sigma.clamp(max=args.sigma_clamp_max))
            writer.add_scalar("sigma/mean", sigma.mean().item(), global_step)
            writer.add_scalar("sigma/min", sigma.min().item(), global_step)
            writer.add_scalar("sigma/max", sigma.max().item(), global_step)
            writer.add_scalar("sigma/log_mean", log_sigma.mean().item(), global_step)
            # W norm diagnostics
            W = agent.actor_mean.weight
            W_row_norms = W.norm(dim=1)
            writer.add_scalar("W_norm/mean", W_row_norms.mean().item(), global_step)
            writer.add_scalar("W_norm/max", W_row_norms.max().item(), global_step)
            # Effective covariance diagonal
            W_dir = W / (W.norm(dim=1, keepdim=True) + 1e-8)
            WS = W_dir.unsqueeze(0) * sigma.unsqueeze(1)
            cov_diag = (WS ** 2).sum(dim=-1)  # [B, ad]
            writer.add_scalar("cov/diag_mean", cov_diag.mean().item(), global_step)
            effective_logstd = 0.5 * cov_diag.mean(dim=0).log()
            writer.add_scalar("losses/effective_logstd_mean", effective_logstd.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
