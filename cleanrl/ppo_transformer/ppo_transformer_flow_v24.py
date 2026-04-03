"""PPO Transformer with Cross-Attentive Directional Exploration (v24).

Goal: let the transformer choose exploration directions and magnitudes directly,
without Cholesky plumbing or autoregressive action-order bias.

Distribution:
  a = mu(s) + U(s) * (sigma_sub(s) ⊙ z) + sigma_diag(s) ⊙ eps
  z ~ N(0, I_K), eps ~ N(0, I_A)

Covariance:
  Sigma(s) = U diag(sigma_sub^2) U^T + diag(sigma_diag^2)

Where cross-attentive transformer-decoded mode tokens produce:
  - U(s): exploration directions (orthonormalized)
  - sigma_sub(s): subspace magnitudes
  - sigma_diag(s): per-dimension residual magnitudes

Exact log_prob is computed with Woodbury + determinant lemma (no approximation).
"""
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
    ent_coef: float = 1e-3
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    log_std_init: float = -0.5
    """initial offset for both diagonal and subspace std heads"""
    use_expln: bool = True
    """use gSDE-style expln positive mapping for std"""
    explore_rank: int = 4
    """rank K of transformer-selected exploration subspace"""
    n_mode_tokens: int = 6
    """number of mode tokens (typically equal to explore_rank)"""
    diag_std_floor: float = 0.08
    """minimum per-dimension residual std"""
    sub_std_floor: float = 0.05
    """minimum per-subspace std"""

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


def _normalize(x, eps=1e-6):
    return F.normalize(x, dim=-1, eps=eps)


def _expln(x, eps=1e-6):
    below = torch.exp(x) * (x <= 0)
    safe_x = x * (x > 0) + eps
    above = (torch.log1p(safe_x) + 1.0) * (x > 0)
    return below + above


class TransformerBlock(nn.Module):
    """nGPT-style block with optional attention mask."""
    def __init__(self, d_token=32, n_heads=4, ffn_hidden=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.d_token = d_token

        self.qkv = nn.Linear(d_token, 3 * d_token, bias=False)
        self.o_proj = nn.Linear(d_token, d_token, bias=False)
        self.ffn_gate_value = nn.Linear(d_token, 2 * ffn_hidden, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, d_token, bias=False)

        self.attn_alpha = nn.Parameter(torch.full((d_token,), 0.05))
        self.ffn_alpha = nn.Parameter(torch.full((d_token,), 0.05))

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(_normalize(x))
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.to(x.dtype).transpose(1, 2).reshape(B, T, D)
        h_attn = _normalize(self.o_proj(attn_out))
        x = _normalize(x + self.attn_alpha * (h_attn - x))

        gate_value = self.ffn_gate_value(_normalize(x))
        gate, value = gate_value.chunk(2, dim=-1)
        h = F.silu(gate) * value
        h_ffn = _normalize(self.ffn_out(h))
        x = _normalize(x + self.ffn_alpha * (h_ffn - x))
        return x


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        hidden_dim = 64
        n_policy_tokens = 8
        d_token = 32

        self._act_dim = int(act_dim)
        self._n_policy_tokens = n_policy_tokens
        self._d_token = d_token
        self._rank = max(1, min(int(args.explore_rank), self._act_dim))
        self._n_mode_tokens = max(self._rank, int(args.n_mode_tokens))

        # === Actor: encoder + mode mixer ===
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_tokenize = nn.Linear(hidden_dim, n_policy_tokens * d_token)
        self.policy_block1 = TransformerBlock(d_token, 4, hidden_dim)
        self.policy_block2 = TransformerBlock(d_token, 4, hidden_dim)

        self.mode_anchor_tokens = nn.Parameter(torch.randn(self._n_mode_tokens, d_token) * 0.02)
        self.mode_tokenize = nn.Linear(hidden_dim, self._n_mode_tokens * d_token)

        # Dedicated exploration decoder: mode queries cross-attend policy tokens,
        # then self-attend among modes to coordinate directional geometry.
        self.mode_heads = 4
        self.mode_head_dim = d_token // self.mode_heads
        self.mode_cross_q = nn.Linear(d_token, d_token, bias=False)
        self.mode_cross_kv = nn.Linear(d_token, 2 * d_token, bias=False)
        self.mode_cross_out = nn.Linear(d_token, d_token, bias=False)
        self.mode_self1 = TransformerBlock(d_token, 4, hidden_dim)
        self.mode_self2 = TransformerBlock(d_token, 4, hidden_dim)

        # Mean + scale heads
        self.mean_agg = layer_init(nn.Linear(d_token, hidden_dim))
        self.mean_norm = RMSNorm(hidden_dim)
        self.mean_out = layer_init(nn.Linear(hidden_dim, self._act_dim), std=0.01)

        self.diag_agg = layer_init(nn.Linear(d_token, hidden_dim))
        self.diag_norm = RMSNorm(hidden_dim)
        self.diag_std_out = nn.Linear(hidden_dim, self._act_dim, bias=True)
        nn.init.zeros_(self.diag_std_out.weight)
        nn.init.zeros_(self.diag_std_out.bias)
        self.diag_log_std_offset = nn.Parameter(torch.full((self._act_dim,), args.log_std_init))

        self.sub_std_out = nn.Linear(d_token, 1, bias=True)
        nn.init.zeros_(self.sub_std_out.weight)
        nn.init.zeros_(self.sub_std_out.bias)
        self.sub_log_std_offset = nn.Parameter(torch.full((self._rank,), args.log_std_init))

        self.dir_out = nn.Linear(d_token, self._act_dim, bias=False)
        nn.init.normal_(self.dir_out.weight, std=0.02)

        self.use_expln = args.use_expln
        self.diag_std_floor = args.diag_std_floor
        self.sub_std_floor = args.sub_std_floor

        # === Critic (separate) ===
        self.critic_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_tokenize = nn.Linear(hidden_dim, n_policy_tokens * d_token)
        self.critic_transformer = TransformerBlock(d_token, 4, hidden_dim)
        self.critic_agg = layer_init(nn.Linear(d_token, hidden_dim))
        self.critic_agg_norm = RMSNorm(hidden_dim)
        self.value_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        self.register_buffer("rank_eye", torch.eye(self._rank))

    def _std_from_raw(self, raw_log_std):
        return _expln(raw_log_std) if self.use_expln else torch.exp(raw_log_std)

    def _get_policy_and_mode_tokens(self, x):
        B = x.shape[0]
        h = F.silu(self.actor_norm1(self.actor_fc1(x)))

        policy_tokens = self.actor_tokenize(h).reshape(B, self._n_policy_tokens, self._d_token)
        policy_tokens = _normalize(policy_tokens)
        policy_tokens = self.policy_block1(policy_tokens)
        policy_tokens = self.policy_block2(policy_tokens)

        mode_state = self.mode_tokenize(h).reshape(B, self._n_mode_tokens, self._d_token)
        mode_anchor = self.mode_anchor_tokens.unsqueeze(0).expand(B, -1, -1)
        mode_tokens = _normalize(mode_state + mode_anchor)

        # mode_queries <-cross-attn- policy_tokens
        q = self.mode_cross_q(mode_tokens).reshape(B, self._n_mode_tokens, self.mode_heads, self.mode_head_dim).transpose(1, 2)
        kv = self.mode_cross_kv(_normalize(policy_tokens))
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, self._n_policy_tokens, self.mode_heads, self.mode_head_dim).transpose(1, 2)
        v = v.reshape(B, self._n_policy_tokens, self.mode_heads, self.mode_head_dim).transpose(1, 2)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            cross_out = F.scaled_dot_product_attention(q, k, v)
        cross_out = cross_out.to(mode_tokens.dtype).transpose(1, 2).reshape(B, self._n_mode_tokens, self._d_token)
        mode_tokens = _normalize(mode_tokens + self.mode_cross_out(cross_out))

        mode_tokens = self.mode_self1(mode_tokens)
        mode_tokens = self.mode_self2(mode_tokens)
        return policy_tokens, mode_tokens

    def _get_dist_params(self, x):
        policy_tokens, mode_tokens = self._get_policy_and_mode_tokens(x)

        pooled = policy_tokens.mean(dim=1)
        mean_h = F.silu(self.mean_norm(self.mean_agg(pooled)))
        mean = self.mean_out(mean_h)

        # Magnitude path is mode-token driven (not pooled policy).
        mode_summary = mode_tokens.mean(dim=1)
        diag_h = F.silu(self.diag_norm(self.diag_agg(mode_summary)))
        diag_raw = self.diag_std_out(diag_h) + self.diag_log_std_offset.unsqueeze(0)
        diag_std = self._std_from_raw(diag_raw) + self.diag_std_floor

        mode_rank = mode_tokens[:, :self._rank]
        sub_raw = self.sub_std_out(mode_rank).squeeze(-1) + self.sub_log_std_offset.unsqueeze(0)
        sub_std = self._std_from_raw(sub_raw) + self.sub_std_floor

        # Mode tokens produce exploration directions; QR gives orthonormal basis.
        U_raw = self.dir_out(mode_rank).transpose(1, 2)  # (B, A, R)
        U, _ = torch.linalg.qr(U_raw, mode="reduced")

        return mean, diag_std, sub_std, U

    def _log_prob_and_entropy(self, delta, diag_std, sub_std, U):
        B = delta.shape[0]
        A = self._act_dim

        d_var = diag_std.pow(2) + 1e-8
        s_var = sub_std.pow(2) + 1e-8
        d_inv = 1.0 / d_var

        term1 = d_inv * delta  # D^{-1} delta
        du = d_inv.unsqueeze(-1) * U
        udu = torch.bmm(U.transpose(1, 2), du)  # U^T D^{-1} U
        s_inv = torch.diag_embed(1.0 / s_var)
        M = s_inv + udu + 1e-6 * self.rank_eye.unsqueeze(0).expand(B, -1, -1)

        rhs = torch.bmm(U.transpose(1, 2), term1.unsqueeze(-1)).squeeze(-1)
        solve = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        corr = d_inv * torch.bmm(U, solve.unsqueeze(-1)).squeeze(-1)
        sigma_inv_delta = term1 - corr
        quad = (delta * sigma_inv_delta).sum(-1)

        logdet_d = torch.log(d_var).sum(-1)
        logdet_s = torch.log(s_var).sum(-1)
        _, logdet_m = torch.linalg.slogdet(M)
        logdet = logdet_d + logdet_s + logdet_m

        const = A * np.log(2 * np.pi)
        log_prob = -0.5 * (quad + const + logdet)
        entropy = 0.5 * (A * (1.0 + np.log(2 * np.pi)) + logdet)
        return log_prob, entropy

    def _critic_features(self, x):
        B = x.shape[0]
        h = F.silu(self.critic_norm1(self.critic_fc1(x)))
        tokens = self.critic_tokenize(h).reshape(B, self._n_policy_tokens, self._d_token)
        tokens = _normalize(tokens)
        tokens = self.critic_transformer(tokens)
        pooled = tokens.mean(dim=1)
        h = F.silu(self.critic_agg_norm(self.critic_agg(pooled)))
        return h

    def get_value(self, x):
        return self.value_out(self._critic_features(x))

    def get_action_and_value(self, x, action=None):
        mean, diag_std, sub_std, U = self._get_dist_params(x)

        if action is None:
            z = torch.randn(mean.shape[0], self._rank, device=x.device, dtype=mean.dtype)
            eps = torch.randn_like(mean)
            sub_noise = torch.bmm(U, (sub_std * z).unsqueeze(-1)).squeeze(-1)
            diag_noise = diag_std * eps
            action = mean + sub_noise + diag_noise

        delta = action - mean
        log_prob, entropy = self._log_prob_and_entropy(delta, diag_std, sub_std, U)
        return action, log_prob, entropy, self.get_value(x)


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
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total parameters: {total_params:,}")

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

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef_low, 1 + args.clip_coef_high)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
        # exploration diagnostics
        with torch.no_grad():
            mean_probe, diag_std_probe, sub_std_probe, U_probe = agent._get_dist_params(b_obs[:256])
            z_probe = torch.randn(256, agent._rank, device=device)
            eps_probe = torch.randn(256, agent._act_dim, device=device)
            sub_noise_probe = torch.bmm(U_probe, (sub_std_probe * z_probe).unsqueeze(-1)).squeeze(-1)
            diag_noise_probe = diag_std_probe * eps_probe
            noise_probe = sub_noise_probe + diag_noise_probe

            utu = torch.bmm(U_probe.transpose(1, 2), U_probe)
            ortho_err = (utu - agent.rank_eye.unsqueeze(0)).pow(2).mean().sqrt()

            writer.add_scalar("explore/noise_std", noise_probe.std().item(), global_step)
            writer.add_scalar("explore/diag_std_mean", diag_std_probe.mean().item(), global_step)
            writer.add_scalar("explore/sub_std_mean", sub_std_probe.mean().item(), global_step)
            writer.add_scalar("explore/sub_std_max", sub_std_probe.max().item(), global_step)
            writer.add_scalar("explore/mean_abs", mean_probe.abs().mean().item(), global_step)
            writer.add_scalar("explore/U_ortho_err", ortho_err.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
