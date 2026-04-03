"""PPO Transformer with Autoregressive Action Generation (v11)

PROFOUND CHANGE: Instead of sampling all action dims from a multivariate
Gaussian (L matrix, coupling flow), generate actions ONE DIM AT A TIME
through the transformer, each conditioned on all previous dims:

  p(a|s) = p(a_0|s) · p(a_1|s,a_0) · p(a_2|s,a_0,a_1) · ...

This is how language models work. It gives STRICTLY MORE EXPRESSIVE distributions
than any multivariate Gaussian — the conditional p(a_i | s, a_{<i}) can capture
arbitrary nonlinear, state-dependent dependencies between action dimensions.

Architecture:
  1. Obs → FC → 8 policy tokens (bidirectional self-attention)
  2. 6 action tokens with causal mask (token i attends to policy + tokens 0..i)
  3. Shifted input: action token i carries a_{i-1}'s value (like LLM teacher forcing)
  4. 2 nGPT transformer blocks with mixed attention mask
  5. Each action token outputs (mean_i, log_std_i) → p(a_i|s,a_{<i}) = N(mean_i, std_i)

Rollout: 6 sequential forward passes (tiny for 14-token transformer)
Training: ONE parallel pass with teacher forcing + causal mask

Log prob = sum_i [-0.5*(eps_i^2 + log(2pi)) - log_std_i]  (exact, trivial)
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
    return F.normalize(x, dim=-1, eps=eps)


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

    def _normalize_weights(self):
        with torch.no_grad():
            for layer in [self.qkv, self.o_proj, self.ffn_gate_value, self.ffn_out]:
                layer.weight.div_(layer.weight.norm(dim=1, keepdim=True).clamp(min=1e-6))

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        self._normalize_weights()

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

        # === Actor encoder: obs → policy tokens ===
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_tokenize = nn.Linear(hidden_dim, n_policy_tokens * d_token)

        # Action dim embeddings (like positional encoding for each action dim)
        self.dim_embeds = nn.Parameter(torch.randn(act_dim, d_token) * 0.02)
        # Value injection: scalar action value → d_token
        self.val_proj = nn.Linear(1, d_token, bias=False)
        nn.init.normal_(self.val_proj.weight, std=0.01)

        # 2 transformer blocks
        self.block1 = TransformerBlock(d_token, 4, hidden_dim)
        self.block2 = TransformerBlock(d_token, 4, hidden_dim)

        # Per-dim readout: action token → (mean, log_std)
        self.action_head = nn.Linear(d_token, 2, bias=True)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

        # === Critic (separate, same as v6) ===
        self.critic_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_tokenize = nn.Linear(hidden_dim, n_policy_tokens * d_token)
        self.critic_transformer = TransformerBlock(d_token, 4, hidden_dim)
        self.critic_agg = layer_init(nn.Linear(d_token, hidden_dim))
        self.critic_agg_norm = RMSNorm(hidden_dim)
        self.value_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        self._n_policy_tokens = n_policy_tokens
        self._d_token = d_token
        self._act_dim = act_dim

        # Precompute causal attention mask
        # Policy tokens: bidirectional among themselves
        # Action tokens: causal (token i sees policy + action 0..i inclusive)
        # Policy tokens do NOT see action tokens
        T = n_policy_tokens + act_dim
        mask = torch.zeros(T, T, dtype=torch.bool)
        mask[:n_policy_tokens, :n_policy_tokens] = True  # policy bidir
        mask[n_policy_tokens:, :n_policy_tokens] = True   # action → policy
        for i in range(act_dim):
            for j in range(i + 1):  # inclusive of self
                mask[n_policy_tokens + i, n_policy_tokens + j] = True
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))  # (1,1,T,T)

    def _get_policy_tokens(self, x):
        B = x.shape[0]
        h = F.silu(self.actor_norm1(self.actor_fc1(x)))
        tokens = self.actor_tokenize(h).reshape(B, self._n_policy_tokens, self._d_token)
        return _normalize(tokens)

    def _build_action_embeds(self, B, action_values, up_to=None):
        """Build shifted action embeddings.

        Position 0: just dim_embed_0 (no previous action)
        Position i>0: dim_embed_i + val_proj(a_{i-1})

        If up_to is given, only inject values for positions 1..up_to.
        """
        act_emb = self.dim_embeds.unsqueeze(0).expand(B, -1, -1).clone()
        if up_to is None:
            up_to = self._act_dim
        # Shifted: position j gets action value j-1
        if up_to > 0 and action_values is not None:
            n = min(up_to, self._act_dim - 1)  # positions 1..n get values 0..n-1
            if n > 0:
                vals = action_values[:, :n].unsqueeze(-1)  # (B, n, 1)
                act_emb[:, 1:n+1] = act_emb[:, 1:n+1] + self.val_proj(vals)
        return _normalize(act_emb)

    def _forward_actor(self, policy_tokens, action_embeds):
        all_tokens = torch.cat([policy_tokens, action_embeds], dim=1)
        all_tokens = self.block1(all_tokens, self.causal_mask)
        all_tokens = self.block2(all_tokens, self.causal_mask)
        return all_tokens[:, self._n_policy_tokens:]  # action outputs only

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
        B = x.shape[0]
        policy_tokens = self._get_policy_tokens(x)

        if action is None:
            # === AUTOREGRESSIVE SAMPLING ===
            sampled = torch.zeros(B, self._act_dim, device=x.device)
            total_lp = torch.zeros(B, device=x.device)

            for i in range(self._act_dim):
                act_emb = self._build_action_embeds(B, sampled, up_to=i)
                action_out = self._forward_actor(policy_tokens, act_emb)

                params_i = self.action_head(action_out[:, i])  # (B, 2)
                mean_i = params_i[:, 0]
                log_std_i = params_i[:, 1].clamp(-5.0, 2.0)

                eps_i = torch.randn_like(mean_i)
                a_i = mean_i + eps_i * log_std_i.exp()
                sampled[:, i] = a_i
                total_lp += -0.5 * (eps_i.pow(2) + np.log(2 * np.pi)) - log_std_i

            action = sampled
            log_prob = total_lp
        else:
            # === TEACHER FORCING (one parallel pass) ===
            act_emb = self._build_action_embeds(B, action, up_to=self._act_dim)
            action_out = self._forward_actor(policy_tokens, act_emb)

            params = self.action_head(action_out)  # (B, act_dim, 2)
            means = params[:, :, 0]
            log_stds = params[:, :, 1].clamp(-5.0, 2.0)

            eps = (action - means) / log_stds.exp()
            log_prob = (-0.5 * (eps.pow(2) + np.log(2 * np.pi)) - log_stds).sum(-1)

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
        # log per-dim std
        with torch.no_grad():
            pt = agent._get_policy_tokens(b_obs[:256])
            ae = agent._build_action_embeds(256, b_actions[:256], up_to=agent._act_dim)
            ao = agent._forward_actor(pt, ae)
            params = agent.action_head(ao)
            log_stds = params[:, :, 1].clamp(-5.0, 2.0)
            writer.add_scalar("explore/log_std_mean", log_stds.mean().item(), global_step)
            writer.add_scalar("explore/log_std_std", log_stds.std().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
