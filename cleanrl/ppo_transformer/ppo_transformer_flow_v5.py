"""PPO Transformer with Attention-Native Correlated Noise (v5)

v5 adds a state-dependent MEAN CORRECTION from the exploration head,
giving the policy dual paths to adjust actions:
  1. Main actor_out: base mean prediction
  2. Attention mean correction: per-actuator refinement via cross+self attention

This mirrors flow v1's shift mechanism, which was the key to its 5200 score.

Architecture (extending v4):
  - Cross-attention + self-attention → L (lower-triangular scale) + mean correction
  - action = (mean + correction) + L @ eps
  - Exact log_prob unchanged: log N(eps;0,I) - sum(log(diag(L)))
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


LOG_STD_INIT = -2.0


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
    """L2-normalize along last dim (project onto unit hypersphere)."""
    return F.normalize(x, dim=-1, eps=eps)


class TransformerBlock(nn.Module):
    """nGPT-style Transformer block: all representations on the unit hypersphere.

    Key ideas from nGPT (NVIDIA, 2024):
    - Hidden states are L2-normalized after each sublayer
    - Weight matrix rows are normalized at each forward pass
    - Residual update: x_new = normalize(x + α * (h - x)) where α is learned
      This is a geodesic interpolation on the hypersphere — bounded step size
    - QK-norm is implicit (q, k are slices of normalized states)
    """
    def __init__(self, d_token=32, n_heads=4, ffn_hidden=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads  # 8
        self.d_token = d_token

        # QKV and output projections
        self.qkv = nn.Linear(d_token, 3 * d_token, bias=False)
        self.o_proj = nn.Linear(d_token, d_token, bias=False)

        # SwiGLU FFN
        self.ffn_gate_value = nn.Linear(d_token, 2 * ffn_hidden, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, d_token, bias=False)

        # nGPT learned step sizes (α), one per sublayer, per-dim
        # Small init → conservative updates at start
        self.attn_alpha = nn.Parameter(torch.full((d_token,), 0.05))
        self.ffn_alpha = nn.Parameter(torch.full((d_token,), 0.05))

    def _normalize_weights(self):
        """Normalize weight rows to unit norm (called each forward)."""
        with torch.no_grad():
            for layer in [self.qkv, self.o_proj, self.ffn_gate_value, self.ffn_out]:
                layer.weight.div_(layer.weight.norm(dim=1, keepdim=True).clamp(min=1e-6))

    def forward(self, x):
        # x: (batch, n_tokens, d_token) — assumed already on unit hypersphere
        B, T, D = x.shape
        self._normalize_weights()

        # --- Multi-head self-attention ---
        qkv = self.qkv(_normalize(x))
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv.unbind(0)
        # Q, K are already unit-scale from normalized input + normalized weights
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.to(x.dtype).transpose(1, 2).reshape(B, T, D)
        h_attn = _normalize(self.o_proj(attn_out))
        # Geodesic step on hypersphere: x + α * (h - x)
        x = _normalize(x + self.attn_alpha * (h_attn - x))

        # --- SwiGLU FFN ---
        gate_value = self.ffn_gate_value(_normalize(x))
        gate, value = gate_value.chunk(2, dim=-1)
        h = F.silu(gate) * value
        h_ffn = _normalize(self.ffn_out(h))
        x = _normalize(x + self.ffn_alpha * (h_ffn - x))

        return x


class ExplorationHead(nn.Module):
    """Attention-native correlated noise via learned scale matrix L.

    Architecture:
    1. Cross-attention: per-actuator queries attend to policy tokens
       → each actuator gathers state-relevant information
    2. Self-attention: actuator tokens attend to each other
       → cross-dim information sharing (all tokens get full context)
    3. Output: each token produces one row of lower-triangular L
       → noise = L @ eps gives cross-dim correlated, state-dependent noise

    Log_prob is exact: log N(eps;0,I) - log|det(L)| = base_lp - sum(log|diag(L)|).
    Inversion is cheap: eps = L^{-1} @ noise via triangular solve.

    Self-attention IS the mechanism for learning cross-dim correlations —
    each actuator's noise scale (its row in L) is informed by what all
    other actuators see in the state.
    """
    def __init__(self, d_token, act_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.d_token = d_token
        self._act_dim = act_dim

        # Learnable exploration queries — one per action dimension
        self.explore_queries = nn.Parameter(torch.randn(act_dim, d_token) * 0.02)

        # Cross-attention: queries attend to policy tokens
        self.cross_q_proj = nn.Linear(d_token, d_token, bias=False)
        self.cross_kv_proj = nn.Linear(d_token, 2 * d_token, bias=False)
        self.cross_out_proj = nn.Linear(d_token, d_token, bias=False)

        # Self-attention: share info across actuator dims
        self.self_qkv = nn.Linear(d_token, 3 * d_token, bias=False)
        self.self_out_proj = nn.Linear(d_token, d_token, bias=False)

        # Output: each token produces act_dim values (one row of L)
        self.L_out = nn.Linear(d_token, act_dim, bias=True)
        nn.init.zeros_(self.L_out.weight)
        # Bias → initial L is diagonal with exp(LOG_STD_INIT) on diagonal
        bias_init = torch.zeros(act_dim)
        self.L_out.bias = nn.Parameter(bias_init)

        # Mean correction: state-dependent shift per actuator (dual-path mean)
        self.mean_corr = nn.Linear(d_token, 1, bias=True)
        nn.init.zeros_(self.mean_corr.weight)
        nn.init.zeros_(self.mean_corr.bias)

        # Pre-computed triangular mask (registered as buffer)
        self.register_buffer('tril_mask', torch.tril(torch.ones(act_dim, act_dim)))
        # Diagonal index mask for separating diagonal from off-diagonal
        self.register_buffer('diag_mask', torch.eye(act_dim))

    def forward(self, policy_tokens):
        """
        Args:
            policy_tokens: (B, n_tokens, d_token) from backbone
        Returns:
            L: (B, act_dim, act_dim) — lower-triangular scale matrix
            log_det: (B,) — log|det(L)| = sum of log|diagonal entries|
            mean_correction: (B, act_dim) — state-dependent mean shift
        """
        B = policy_tokens.shape[0]
        A = self._act_dim

        # --- Cross-attention: gather per-actuator state info ---
        q = self.cross_q_proj(self.explore_queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.cross_kv_proj(policy_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, A, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            cross_out = F.scaled_dot_product_attention(q, k, v)
        x = cross_out.to(policy_tokens.dtype).transpose(1, 2).reshape(B, A, self.d_token)
        x = self.cross_out_proj(x)  # (B, act_dim, d_token)

        # --- Self-attention: cross-dim information sharing ---
        qkv = self.self_qkv(x).reshape(B, A, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        sq, sk, sv = qkv.unbind(0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            self_out = F.scaled_dot_product_attention(sq, sk, sv)
        self_out = self_out.to(x.dtype).transpose(1, 2).reshape(B, A, self.d_token)
        x = x + self.self_out_proj(self_out)  # residual

        # --- Produce lower-triangular scale matrix L ---
        L_raw = self.L_out(x)  # (B, act_dim, act_dim)

        # Apply triangular mask (zero out upper triangle)
        L = L_raw * self.tril_mask

        # Make diagonal positive via exp, clamp for stability
        diag = torch.diagonal(L, dim1=-2, dim2=-1)  # (B, act_dim)
        log_diag = diag.clamp(-5.0, 2.0)  # raw diagonal → log scale
        pos_diag = torch.exp(log_diag)

        # Replace diagonal: L = L_offdiag + diag_embed(pos_diag)
        L = L * (1 - self.diag_mask) + torch.diag_embed(pos_diag)

        # log|det(L)| = sum of log of positive diagonal entries
        log_det = log_diag.sum(-1)

        # Mean correction from enriched per-actuator features
        mean_correction = self.mean_corr(x).squeeze(-1)  # (B, act_dim)

        return L, log_det, mean_correction


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        hidden_dim = 64
        n_tokens = 8
        d_token = 32

        # === Actor backbone: obs → transformer → token representations ===
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_tokenize = nn.Linear(hidden_dim, n_tokens * d_token)
        self.actor_transformer = TransformerBlock(d_token=d_token, n_heads=4, ffn_hidden=hidden_dim)

        # Mean head: pool tokens → project to actions
        self.actor_agg = layer_init(nn.Linear(d_token, hidden_dim))
        self.actor_agg_norm = RMSNorm(hidden_dim)
        self.actor_out = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

        # Exploration head: cross-attn + self-attn over policy tokens
        self.explore_head = ExplorationHead(d_token, act_dim, n_heads=4)

        # === Critic backbone ===
        self.critic_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_tokenize = nn.Linear(hidden_dim, n_tokens * d_token)
        self.critic_transformer = TransformerBlock(d_token=d_token, n_heads=4, ffn_hidden=hidden_dim)
        self.critic_agg = layer_init(nn.Linear(d_token, hidden_dim))
        self.critic_agg_norm = RMSNorm(hidden_dim)
        self.value_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        self._n_tokens = n_tokens
        self._d_token = d_token

    def _get_policy_tokens(self, x):
        B = x.shape[0]
        h = F.silu(self.actor_norm1(self.actor_fc1(x)))
        tokens = self.actor_tokenize(h).reshape(B, self._n_tokens, self._d_token)
        tokens = _normalize(tokens)
        tokens = self.actor_transformer(tokens)
        return tokens

    def _get_mean(self, tokens):
        pooled = tokens.mean(dim=1)
        h = F.silu(self.actor_agg_norm(self.actor_agg(pooled)))
        return self.actor_out(h)

    def _critic_features(self, x):
        B = x.shape[0]
        h = F.silu(self.critic_norm1(self.critic_fc1(x)))
        tokens = self.critic_tokenize(h).reshape(B, self._n_tokens, self._d_token)
        tokens = _normalize(tokens)
        tokens = self.critic_transformer(tokens)
        pooled = tokens.mean(dim=1)
        h = F.silu(self.critic_agg_norm(self.critic_agg(pooled)))
        return h

    def get_value(self, x):
        return self.value_out(self._critic_features(x))

    def get_action_and_value(self, x, action=None):
        tokens = self._get_policy_tokens(x)
        base_mean = self._get_mean(tokens)

        # ExplorationHead: L (noise scale) + mean correction (dual-path mean)
        L, log_det, mean_correction = self.explore_head(tokens)
        mean = base_mean + mean_correction

        if action is None:
            # Sample: noise = L @ eps, action = mean + noise
            eps = torch.randn_like(mean)
            noise = torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
            action = mean + noise
        else:
            # PPO update: recover eps from stored action
            # noise = action - mean, eps = L^{-1} @ noise
            noise = action - mean
            eps = torch.linalg.solve_triangular(
                L, noise.unsqueeze(-1), upper=False
            ).squeeze(-1)

        # Exact log_prob: log N(eps; 0, I) - log|det(L)|
        base_lp = -0.5 * (eps.pow(2) + np.log(2 * np.pi)).sum(-1)
        log_prob = base_lp - log_det

        entropy = -log_prob.detach()
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

    agent = Agent(envs, args).to(device)
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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
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
        # noise metrics
        with torch.no_grad():
            tokens_probe = agent._get_policy_tokens(b_obs[:256])
            L_probe, log_det_probe, mcorr_probe = agent.explore_head(tokens_probe)
            eps_probe = torch.randn(256, np.prod(envs.single_action_space.shape), device=device)
            noise_probe = torch.bmm(L_probe, eps_probe.unsqueeze(-1)).squeeze(-1)
            diag_probe = torch.diagonal(L_probe, dim1=-2, dim2=-1)
            offdiag_norm = (L_probe * (1 - agent.explore_head.diag_mask)).pow(2).sum((-2, -1)).sqrt()
            writer.add_scalar("explore/noise_std", noise_probe.std().item(), global_step)
            writer.add_scalar("explore/noise_mean_abs", noise_probe.abs().mean().item(), global_step)
            writer.add_scalar("explore/log_det_mean", log_det_probe.mean().item(), global_step)
            writer.add_scalar("explore/diag_mean", diag_probe.mean().item(), global_step)
            writer.add_scalar("explore/offdiag_norm", offdiag_norm.mean().item(), global_step)
            writer.add_scalar("explore/mean_corr_abs", mcorr_probe.abs().mean().item(), global_step)
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
