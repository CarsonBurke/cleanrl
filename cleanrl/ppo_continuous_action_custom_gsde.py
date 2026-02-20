# QL-SDE v24d: Advantage-Guided RPO
#
# Builds on v22b (best: HC 6,079 final / 6,258 peak).
#
# Key insight from v24a-c: pure gradient-based noise floor self-extinguishes
# as the advantage landscape flattens near the optimal mean. RPO provides
# constant exploration pressure (good) but applies it uniformly (wasteful).
#
# This version combines both:
#   - RPO controls the exploration BUDGET (constant alpha)
#   - Advantage predictor gradient controls the ALLOCATION (which dims get more)
#
# During training, RPO noise is modulated per-dimension:
#   grad = |∇_a A_pred(obs, mean)|  (detached from policy)
#   dim_weights = grad / mean(grad)  (normalize to mean=1)
#   dim_weights = clamp(dim_weights, 0.2, 5.0)  (bound modulation)
#   noise = uniform(-alpha, alpha) * dim_weights
#   mean_train = mean + noise
#
# The advantage predictor is trained on raw GAE advantages via MSE.
# Early on (predictor untrained), weights ≈ 1.0 → behaves like standard RPO.
# As predictor learns, noise concentrates on high-sensitivity dimensions.
#
# v22b baseline: HC=6,079 (with RPO α=0.1)
# v22b no-RPO:   HC=4,072 (policy collapses without exploration help)
# v24c (noise floor only): HC≈4,500 (noise floor self-extinguishes)

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
    """RPO noise alpha — base exploration magnitude"""

    # QL-SDE specific arguments
    sde_dim: int = 64
    """dimensionality of the SDE latent space"""

    # Advantage predictor arguments
    adv_pred_hidden: int = 64
    """hidden dimension for advantage predictor network"""
    adv_pred_coef: float = 0.5
    """loss coefficient for advantage predictor MSE"""
    adv_mod_min: float = 0.2
    """minimum per-dimension modulation weight"""
    adv_mod_max: float = 5.0
    """maximum per-dimension modulation weight"""

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


class AdvantagePredictor(nn.Module):
    """Predicts advantage A(obs, action) → scalar.

    Gradient ∇_a A provides per-dimension action sensitivity used to
    modulate RPO noise allocation across dimensions.
    """
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Standard init for meaningful gradients from the start
        nn.init.orthogonal_(self.net[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.net[0].bias)
        nn.init.orthogonal_(self.net[2].weight, gain=np.sqrt(2))
        nn.init.zeros_(self.net[2].bias)
        nn.init.orthogonal_(self.net[4].weight, gain=0.1)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class Agent(nn.Module):
    def __init__(self, envs, sde_dim=64, rpo_alpha=0.1,
                 adv_pred_hidden=64, adv_mod_min=0.2, adv_mod_max=5.0):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.sde_dim = sde_dim
        self.rpo_alpha = rpo_alpha
        self.adv_mod_min = adv_mod_min
        self.adv_mod_max = adv_mod_max
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

        # Advantage predictor (separate from policy graph)
        self.adv_predictor = AdvantagePredictor(obs_dim, action_dim, hidden=adv_pred_hidden)

    def get_value(self, x):
        return self.critic(x)

    def _compute_dim_weights(self, obs, action_mean):
        """Compute per-dimension RPO modulation weights from advantage gradient.

        Returns: dim_weights [B, action_dim] with mean≈1, clamped to [mod_min, mod_max].
        Fully detached from policy graph.
        """
        obs_d = obs.detach()
        act_d = action_mean.detach().requires_grad_(True)

        with torch.enable_grad():
            adv_pred = self.adv_predictor(obs_d, act_d)
            grad = torch.autograd.grad(
                adv_pred.sum(), act_d, create_graph=False
            )[0]  # [B, action_dim]

        # Normalize |∇_a A| to mean=1 per sample, then clamp
        grad_abs = grad.abs()
        grad_mean = grad_abs.mean(dim=-1, keepdim=True) + 1e-8
        dim_weights = (grad_abs / grad_mean).clamp(self.adv_mod_min, self.adv_mod_max)

        return dim_weights.detach()

    def get_action_and_value(self, x, action=None):
        B = x.shape[0]
        ad = self.action_dim

        # Mean path
        mean_latent = self.actor_mean_trunk(x)
        action_mean = self.actor_mean(mean_latent)

        # Covariance path → Gram-tanh Cholesky (pure v22b)
        cov_latent = self.actor_cov_trunk(x)
        sde_raw = self.sde_fc(cov_latent)
        sde_latent = sde_raw.view(B, ad, self.sde_dim)
        L = torch.tanh(sde_latent) * (1.0 / (self.sde_dim ** 0.5))
        cov = torch.bmm(L, L.transpose(1, 2))  # [B, ad, ad]
        eps_eye = 1e-4 * torch.eye(ad, device=x.device)
        cov = cov + eps_eye
        chol = torch.linalg.cholesky(cov)

        # Advantage-guided RPO: modulated noise on mean (training only)
        dim_weights = None
        if action is not None and self.rpo_alpha > 0:
            dim_weights = self._compute_dim_weights(x, action_mean)  # [B, ad]
            z_rpo = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z_rpo * dim_weights

        if action is None:
            z = torch.randn(B, ad, 1, device=x.device)
            action = action_mean + torch.bmm(chol, z).squeeze(-1)

        # Log prob via Cholesky solve
        diff = (action - action_mean).unsqueeze(-1)
        solved = torch.linalg.solve_triangular(chol, diff, upper=False)
        mahal = solved.squeeze(-1).pow(2).sum(dim=-1)
        log_diag = chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        log_prob = -0.5 * (ad * self._log2pi + 2 * log_diag + mahal)

        # Entropy
        entropy = 0.5 * (ad * (1.0 + self._log2pi) + 2 * log_diag)

        return action, log_prob, entropy, self.critic(x), dim_weights


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
        adv_pred_hidden=args.adv_pred_hidden,
        adv_mod_min=args.adv_mod_min,
        adv_mod_max=args.adv_mod_max,
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
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
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
        adv_pred_losses = []
        dim_weight_stds = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, dim_weights = agent.get_action_and_value(
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

                # Advantage predictor loss (trains on raw unnormalized advantages)
                adv_pred = agent.adv_predictor(b_obs[mb_inds], b_actions[mb_inds])
                adv_pred_loss = F.mse_loss(adv_pred, b_advantages[mb_inds].detach())

                entropy_loss = entropy.mean()
                loss = (pg_loss
                        - args.ent_coef * entropy_loss
                        + v_loss * args.vf_coef
                        + args.adv_pred_coef * adv_pred_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Track diagnostics
                with torch.no_grad():
                    adv_pred_losses.append(adv_pred_loss.item())
                    if dim_weights is not None:
                        dim_weight_stds.append(dim_weights.std().item())

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
        # Advantage predictor diagnostics
        writer.add_scalar("losses/adv_pred_loss", np.mean(adv_pred_losses), global_step)
        if dim_weight_stds:
            writer.add_scalar("adv_pred/dim_weight_std", np.mean(dim_weight_stds), global_step)
        with torch.no_grad():
            # Log effective covariance magnitude
            cov_lat = agent.actor_cov_trunk(b_obs[mb_inds])
            ad = agent.action_dim
            sde_raw = agent.sde_fc(cov_lat)
            sde_latent = sde_raw.view(-1, ad, agent.sde_dim)
            L_val = torch.tanh(sde_latent) * (1.0 / (agent.sde_dim ** 0.5))
            cov_val = torch.bmm(L_val, L_val.transpose(1, 2)) + 1e-4 * torch.eye(ad, device=device)
            chol_val = torch.linalg.cholesky(cov_val)
            log_diag = chol_val.diagonal(dim1=-2, dim2=-1).log()
            writer.add_scalar("losses/effective_logstd_mean", log_diag.mean().item(), global_step)
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
