# QL-SDE v23: Gram-Tanh Covariance + Conditional Normalizing Flow
#
# Builds on v22b (best: HC 6,079 final / 6,258 peak) by adding a state-dependent
# normalizing flow after the Cholesky correlation step. This transforms Gaussian
# noise through learned invertible nonlinear layers, enabling:
#   - State-dependent kurtosis (heavy tails when uncertain, tight for fine control)
#   - Skewness (asymmetric exploration toward promising directions)
#   - Non-Gaussian exploration shape that adapts per-state
#
# Architecture:
#   z ~ N(0, I) → chol(obs) @ z → CouplingFlow(·; obs) → + mean(obs) → action
#
# The flow is a 2-layer affine coupling (RealNVP-style):
#   Layer 1: transform second-half dims conditioned on (first-half, obs)
#   Layer 2: transform first-half dims conditioned on (transformed-second-half, obs)
#
# Log-prob remains exact: log N(z;0,I) - log|det(chol)| - sum(s1) - sum(s2)
# Zero-initialized flow → starts as identity → model begins as pure MVN

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
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RPO
    rpo_alpha: float = 0.1
    """RPO noise alpha — adds bounded uniform noise to action mean during training"""

    # QL-SDE specific arguments
    sde_dim: int = 64
    """dimensionality of the SDE latent space"""

    # Flow specific arguments
    flow_hidden: int = 32
    """hidden dimension for coupling flow networks"""

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


# ---------------------------------------------------------------------------
# Conditional Normalizing Flow (RealNVP-style affine coupling)
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """Affine coupling: transforms x_transform conditioned on (x_cond, obs)."""

    def __init__(self, cond_dim, transform_dim, obs_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * transform_dim),
        )
        # Zero-init last layer → identity transform at initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _get_scale_shift(self, x_cond, obs):
        st = self.net(torch.cat([x_cond, obs], dim=-1))
        s_raw, t = st.chunk(2, dim=-1)
        s = torch.tanh(s_raw) * 2.0  # clamp: exp(s) in [0.14, 7.39]
        return s, t

    def forward(self, x_cond, x_transform, obs):
        """Forward: x_transform → y, returns (y, log_scale)."""
        s, t = self._get_scale_shift(x_cond, obs)
        y = x_transform * torch.exp(s) + t
        return y, s

    def inverse(self, x_cond, y, obs):
        """Inverse: y → x_transform, returns (x_transform, log_scale)."""
        s, t = self._get_scale_shift(x_cond, obs)
        x = (y - t) * torch.exp(-s)
        return x, s


class CouplingFlow(nn.Module):
    """Two-layer affine coupling flow conditioned on observation.

    Layer 1: transforms dims [d1:] conditioned on (dims [:d1], obs)
    Layer 2: transforms dims [:d1] conditioned on (transformed dims [d1:], obs)
    """

    def __init__(self, action_dim, obs_dim, hidden=32):
        super().__init__()
        self.d1 = action_dim // 2
        self.d2 = action_dim - self.d1
        self.layer1 = AffineCouplingLayer(self.d1, self.d2, obs_dim, hidden)
        self.layer2 = AffineCouplingLayer(self.d2, self.d1, obs_dim, hidden)

    def forward(self, x, obs):
        """Forward: x → y, returns (y, log_det_jacobian)."""
        x1, x2 = x[..., :self.d1], x[..., self.d1:]
        h2, s1 = self.layer1.forward(x1, x2, obs)
        h1, s2 = self.layer2.forward(h2, x1, obs)
        y = torch.cat([h1, h2], dim=-1)
        log_det = s1.sum(-1) + s2.sum(-1)
        return y, log_det

    def inverse(self, y, obs):
        """Inverse: y → x, returns (x, forward_log_det_jacobian)."""
        h1, h2 = y[..., :self.d1], y[..., self.d1:]
        x1, s2 = self.layer2.inverse(h2, h1, obs)
        x2, s1 = self.layer1.inverse(x1, h2, obs)
        x = torch.cat([x1, x2], dim=-1)
        log_det = s1.sum(-1) + s2.sum(-1)
        return x, log_det


class Agent(nn.Module):
    def __init__(self, envs, sde_dim=64, rpo_alpha=0.1, flow_hidden=32):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.action_dim = action_dim
        self.sde_dim = sde_dim
        self.rpo_alpha = rpo_alpha
        self._log2pi = np.log(2 * np.pi)

        # Critic: scalar value head with SiLU+RMSNorm
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            nn.RMSNorm(64),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor mean trunk (separate from covariance)
        self.actor_mean_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)

        # Covariance trunk (separate pathway)
        self.actor_cov_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            nn.RMSNorm(64),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            nn.RMSNorm(64),
        )
        self.sde_fc = layer_init(nn.Linear(64, action_dim * sde_dim), std=1.0)

        # Conditional normalizing flow (after Cholesky correlation)
        self.flow = CouplingFlow(action_dim, obs_dim, hidden=flow_hidden)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        B = x.shape[0]
        ad = self.action_dim

        # Mean path
        mean_latent = self.actor_mean_trunk(x)
        action_mean = self.actor_mean(mean_latent)

        # Covariance path → Cholesky
        cov_latent = self.actor_cov_trunk(x)
        sde_raw = self.sde_fc(cov_latent)
        sde_latent = sde_raw.view(B, ad, self.sde_dim)
        L = torch.tanh(sde_latent) * (1.0 / (self.sde_dim ** 0.5))
        cov = torch.bmm(L, L.transpose(1, 2))
        eps_eye = 1e-4 * torch.eye(ad, device=x.device)
        cov = cov + eps_eye
        chol = torch.linalg.cholesky(cov)

        # RPO: bounded uniform noise on mean (training only)
        if action is not None and self.rpo_alpha > 0:
            z_rpo = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z_rpo

        log_diag = chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [B]

        if action is None:
            # Forward: z → chol @ z → flow → + mean
            z = torch.randn(B, ad, 1, device=x.device)
            correlated = torch.bmm(chol, z).squeeze(-1)          # [B, ad]
            noise, flow_log_det = self.flow.forward(correlated, x)
            action = action_mean + noise
            mahal = z.squeeze(-1).pow(2).sum(dim=-1)              # [B]
        else:
            # Inverse: action → noise → flow⁻¹ → correlated → chol⁻¹ → z
            noise = action - action_mean
            correlated, flow_log_det = self.flow.inverse(noise, x)
            z_col = torch.linalg.solve_triangular(
                chol, correlated.unsqueeze(-1), upper=False
            )
            mahal = z_col.squeeze(-1).pow(2).sum(dim=-1)          # [B]

        # log p(action) = log N(z;0,I) - log|det(chol)| - flow_log_det
        log_prob = -0.5 * (ad * self._log2pi + mahal) - log_diag - flow_log_det

        # Entropy estimate: base Gaussian entropy + Jacobian volume change
        # Exact for the Gaussian part; point estimate for flow contribution
        entropy = 0.5 * (ad * (1.0 + self._log2pi)) + log_diag + flow_log_det

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

    agent = Agent(
        envs,
        sde_dim=args.sde_dim,
        rpo_alpha=args.rpo_alpha,
        flow_hidden=args.flow_hidden,
    ).to(device)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        with torch.no_grad():
            # Log effective covariance magnitude
            cov_lat = agent.actor_cov_trunk(b_obs[mb_inds])
            ad = agent.action_dim
            sde_raw = agent.sde_fc(cov_lat)
            sde_latent = sde_raw.view(-1, ad, agent.sde_dim)
            L_diag = torch.tanh(sde_latent) * (1.0 / (agent.sde_dim ** 0.5))
            cov_diag = torch.bmm(L_diag, L_diag.transpose(1, 2)) + 1e-4 * torch.eye(ad, device=device)
            chol_diag = torch.linalg.cholesky(cov_diag)
            log_diag_vals = chol_diag.diagonal(dim1=-2, dim2=-1).log()
            writer.add_scalar("losses/effective_logstd_mean", log_diag_vals.mean().item(), global_step)
            # Log flow contribution: forward pass on fresh noise to measure log_det
            z_probe = torch.randn(b_obs[mb_inds].shape[0], ad, 1, device=device)
            corr_probe = torch.bmm(chol_diag, z_probe).squeeze(-1)
            _, flow_ld = agent.flow.forward(corr_probe, b_obs[mb_inds])
            writer.add_scalar("losses/flow_log_det_mean", flow_ld.mean().item(), global_step)
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
