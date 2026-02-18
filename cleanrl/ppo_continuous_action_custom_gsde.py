# QL-SDE + DreamerV3-style symexp-twohot distributional critic
#
# Base: custom_gsde v23 (direct Cholesky parameterization)
# Addition: Replace scalar critic with symexp-twohot distributional value head
#   - Critic outputs logits over symexp-spaced bins instead of a single scalar
#   - Value loss: cross-entropy on two-hot encoded returns (not MSE)
#   - Prediction: softmax-weighted sum over bin centres
#
# BUCKET RANGE NOTE:
#   DreamerV3 defaults to linspace(-20, 0) → symexp covers ±4.85e8.
#   With NormalizeReward + clip(-10, 10), practical returns are O(10-100).
#   We use bin_range=8 → symexp(8)≈2981, covering the realistic range
#   with much denser bucket spacing. This avoids wasting 99.9% of bins
#   on unreachable values.
#
# ARCHITECTURE:
#   Actor: same as custom_gsde (shared trunk → mean + direct Cholesky)
#   Critic: trunk → Linear(num_bins) → softmax → weighted sum of symexp bins
#   Value loss: cross-entropy(logits, twohot_encode(returns))

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


# ---------------------------------------------------------------------------
# DreamerV3 primitives (following reference impl at ../dreamerv3)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exp (inverse of symlog): sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def build_symexp_bins(num_bins: int, bin_range: float = 8.0) -> torch.Tensor:
    """Build symmetric bin centres in symexp-space.

    Bins are linearly spaced in symlog-space from [-bin_range, +bin_range],
    then transformed via symexp to get non-uniform spacing in value-space
    (dense near 0, coarse at extremes).

    bin_range controls the max representable value: symexp(bin_range).
    Default 8.0 → max ≈ ±2981, suitable for normalized-reward envs.
    """
    if num_bins % 2 == 1:
        half = torch.linspace(-bin_range, 0.0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(0)])
    else:
        half = torch.linspace(-bin_range, 0.0, num_bins // 2)
        half = symexp(half)
        bins = torch.cat([half, -half.flip(0)])
    return bins


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets as two-hot vectors over bins."""
    x = x.unsqueeze(-1)  # (..., 1)
    below = (bins <= x).long().sum(-1) - 1
    below = below.clamp(0, len(bins) - 1)
    above = (below + 1).clamp(0, len(bins) - 1)
    equal = (below == above)
    dist_below = torch.where(equal, torch.ones_like(x.squeeze(-1)),
                             torch.abs(bins[below] - x.squeeze(-1)))
    dist_above = torch.where(equal, torch.ones_like(x.squeeze(-1)),
                             torch.abs(bins[above] - x.squeeze(-1)))
    total = dist_below + dist_above
    weight_below = dist_above / total
    weight_above = dist_below / total
    target = (F.one_hot(below, len(bins)).float() * weight_below.unsqueeze(-1)
              + F.one_hot(above, len(bins)).float() * weight_above.unsqueeze(-1))
    return target


def twohot_predict(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode predicted value from two-hot logits via symmetric weighted sum.

    Pairs negative/positive bins to cancel floating-point error, ensuring
    prediction is exactly zero when logits are uniform at initialization.
    """
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1 = probs[..., :m]
        p2 = probs[..., m : m + 1]
        p3 = probs[..., m + 1 :]
        b1 = bins[:m]
        b2 = bins[m : m + 1]
        b3 = bins[m + 1 :]
        return (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
    else:
        p1 = probs[..., : n // 2]
        p2 = probs[..., n // 2 :]
        b1 = bins[: n // 2]
        b2 = bins[n // 2 :]
        return ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)


def twohot_loss(logits: torch.Tensor, target_twohot: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss against a two-hot target (per-element, unreduced)."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -(target_twohot * log_probs).sum(-1)


# ---------------------------------------------------------------------------
# Args & env helpers
# ---------------------------------------------------------------------------

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
    clip_vloss: bool = False
    """Toggles whether to use a clipped loss for the value function (off by default — twohot CE doesn't need it)."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RPO
    rpo_alpha: float = 0.0
    """RPO noise alpha — adds bounded uniform noise to action mean during training"""

    # QL-SDE specific arguments
    sde_dim: int = 64
    """dimensionality of the SDE latent space (Gram factor width)"""

    # Twohot critic arguments
    num_bins: int = 255
    """number of bins for symexp-twohot value head"""
    bin_range: float = 8.0
    """symlog-space range for bins (symexp(bin_range) = max representable value)"""

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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent(nn.Module):
    def __init__(self, envs, sde_dim=64, rpo_alpha=0.05, num_bins=255, bin_range=8.0):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.action_dim = action_dim
        self.rpo_alpha = rpo_alpha
        self._log2pi = np.log(2 * np.pi)

        # Critic: outputs logits over symexp-twohot bins instead of a scalar
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.RMSNorm(64),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, num_bins), std=0.01),
        )

        # Register symexp-twohot bin centres as a buffer (not a parameter)
        self.register_buffer("bins", build_symexp_bins(num_bins, bin_range))

        # Actor mean trunk
        self.actor_mean_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.RMSNorm(64),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)

        # Covariance trunk
        self.actor_cov_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.RMSNorm(64),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
        )

        # Gram factor: projects to L [ad, sde_dim], then cov = L@L^T / sde_dim + εI
        self.sde_dim = sde_dim
        self.sde_fc = layer_init(nn.Linear(64, action_dim * sde_dim), std=0.01)
        self.sde_norm = nn.RMSNorm(action_dim * sde_dim)

    def get_value(self, x):
        """Return scalar value predictions (symexp-twohot decoded)."""
        logits = self.critic(x)
        return twohot_predict(logits, self.bins)

    def get_action_and_value(self, x, action=None):
        B = x.shape[0]
        ad = self.action_dim

        # Mean path (own trunk)
        mean_latent = self.actor_mean_trunk(x)             # [B, 64]
        action_mean = self.actor_mean(mean_latent)         # [B, ad]

        # Covariance path: trunk → sde_fc → RMSNorm → 2*softsign bounds L to [-2,2]
        cov_latent = self.actor_cov_trunk(x)               # [B, 64]
        L_raw = self.sde_norm(self.sde_fc(cov_latent))     # [B, ad*k] normalized
        L_raw = L_raw.view(B, ad, self.sde_dim)            # [B, ad, k]
        L = 2.0 * L_raw / (1.0 + L_raw.abs())             # [B, ad, k]

        # Gram covariance: Σ = L@L^T / k + εI
        cov = torch.bmm(L, L.transpose(1, 2)) / self.sde_dim       # [B, ad, ad]
        cov.diagonal(dim1=-2, dim2=-1).add_(1e-6)
        chol = torch.linalg.cholesky(cov)                  # [B, ad, ad]

        # RPO: add bounded uniform noise to mean during training only
        if action is not None and self.rpo_alpha > 0:
            z = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z

        if action is None:
            # Sample: action = mean + chol @ z
            z = torch.randn(B, ad, 1, device=x.device)
            action = action_mean + torch.bmm(chol, z).squeeze(-1)

        # Log prob via Cholesky: -0.5 * (k*log(2π) + 2*sum(log(diag(chol))) + ||chol⁻¹(x-μ)||²)
        diff = (action - action_mean).unsqueeze(-1)        # [B, ad, 1]
        solved = torch.linalg.solve_triangular(chol, diff, upper=False)  # [B, ad, 1]
        mahal = solved.squeeze(-1).pow(2).sum(dim=-1)      # [B]
        log_diag = chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [B]
        log_prob = -0.5 * (ad * self._log2pi + 2 * log_diag + mahal)  # [B]

        # Entropy: 0.5 * (k * (1 + log(2π)) + 2 * sum(log(diag(chol))))
        entropy = 0.5 * (ad * (1.0 + self._log2pi) + 2 * log_diag)  # [B]

        # Critic: twohot logits + decoded scalar
        value_logits = self.critic(x)
        value = twohot_predict(value_logits, self.bins)

        return action, log_prob, entropy, value, value_logits


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
        rpo_alpha=args.rpo_alpha,
        num_bins=args.num_bins,
        bin_range=args.bin_range,
    ).to(device)

    # Separate param groups: 10x LR for covariance path to compensate Gram gradient dilution
    cov_params = set(agent.actor_cov_trunk.parameters()) | set(agent.sde_fc.parameters())
    other_params = [p for p in agent.parameters() if p not in cov_params]
    optimizer = optim.Adam([
        {"params": other_params},
        {"params": list(cov_params), "lr": args.learning_rate * 10},
    ], lr=args.learning_rate, eps=1e-5)

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
            optimizer.param_groups[1]["lr"] = lrnow * 10

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
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

                _, newlogprob, entropy, newvalue, newvalue_logits = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
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

                # Value loss: symexp-twohot cross-entropy
                mb_returns = b_returns[mb_inds]
                target_twohot = twohot_encode(mb_returns, agent.bins)
                v_loss = twohot_loss(newvalue_logits, target_twohot).mean()

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
        with torch.no_grad():
            cov_lat = agent.actor_cov_trunk(b_obs[mb_inds])
            ad = agent.action_dim
            Bm = cov_lat.shape[0]
            L_raw_log = agent.sde_norm(agent.sde_fc(cov_lat)).view(Bm, ad, agent.sde_dim)
            L_log = 2.0 * L_raw_log / (1.0 + L_raw_log.abs())
            cov_log = torch.bmm(L_log, L_log.transpose(1, 2)) / agent.sde_dim
            cov_log.diagonal(dim1=-2, dim2=-1).add_(1e-6)
            chol_log = torch.linalg.cholesky(cov_log)
            log_diag = chol_log.diagonal(dim1=-2, dim2=-1).log()
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
