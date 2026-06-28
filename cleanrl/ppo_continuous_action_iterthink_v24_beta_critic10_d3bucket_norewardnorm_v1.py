# PPO + IterThink v24 BETA — CRITIC-10 + Dreamer3-bucket HL-Gauss critic, NO reward-norm v1.
#
# Sibling of iterthink_v24_beta_critic10_hlgauss_sepbb_v1. Same iterthink_v24 beta
# ARCHITECTURE (DenseNet ThinkTrunk + soft-MoE, unimodal Beta policy) and the same
# ASYMMETRIC schedule (ACTOR 1 epoch / CRITIC 10 epochs, no ratio clip, no KL stop,
# NO advantage normalization, separate actor/critic trunks). This variant changes only
# TWO axes vs that run:
#
#   (1) NO REWARD NORMALIZATION. The NormalizeReward + reward-clip wrappers are removed
#       from make_env, so the critic regresses RAW environment returns (HalfCheetah
#       lambda-returns reach the thousands). This kills the reward-norm late-training
#       advantage-shrinkage failure mode (scale compresses as the policy improves with
#       no adv-norm to re-inflate it) at the cost of training on a non-stationary,
#       large-magnitude return target — which is exactly what the wide symlog critic
#       below is built to absorb.
#
#   (2) DREAMER3-BUCKET HL-GAUSS CRITIC (ported from iterthink_v166's value head; the
#       symlog scaling, bucket spacing, coordinate range, and decoding ONLY — NO MTP /
#       world-model loss, NO distributional bootstrap). Instead of the symlog *linear*
#       HL-Gauss support, the support is built in SYMLOG-COORDINATE space and the raw
#       bucket centers are symexp(linspace(v_min, v_max, num_bins)) — i.e. Dreamer3
#       exponentially-spaced buckets, dense near 0 and sparse out to ±symexp(9.9035)≈
#       ±19999 raw, with one exact zero bucket (num_bins=511 odd). Targets are the
#       HL-Gauss Gaussian-CDF projection of the SCALAR GAE lambda-return onto the
#       symlog-coordinate intervals (clamped to the coord support), sigma_ratio=0.75 of
#       the coordinate bin width; the value is decoded as E[symexp(center)]. This covers
#       any raw-return scale with fine resolution near 0, which is what makes removing
#       reward-norm safe.
#
# UNCHANGED FROM THE critic10 LINE:
#   - ThinkTrunk (entry -> K dense-concat ThinkBlocks {dense + softmax-MoE} -> proj).
#   - Unimodal Beta actor + z-replay; constant rescale Jacobian drops out of the ratio.
#   - ACTOR 1 / CRITIC 10 epochs in two decoupled loops; clip_ratio=False (plain -A*ratio);
#     norm_adv=False (raw GAE); target_kl=None; ent_coef=0; max_grad_norm=0.5; num_steps=2048.
#   - share_backbone=False default (separate actor/critic trunks), matching the sepbb run.
#
# HYPOTHESIS: regressing RAW returns with a Dreamer3-bucket symlog critic removes the
# reward-norm scale-compression pathology while keeping advantages well-calibrated, so
# the single-pass unclipped PG actor sees stable, correctly-scaled advantages for the
# whole 8M-step run. Bar: the reward-normed sepbb sibling and the iterthink line.
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 1            # ONE faithful on-policy actor pass per rollout
    critic_epochs: int = 10          # refit the critic many times (pure distributional regression)
    norm_adv: bool = False           # NO adv norm (per-minibatch standardize OFF)
    clip_ratio: bool = False         # NO PPO ratio clip: plain -A*ratio surrogate (clamp OFF)
    clip_coef: float = 0.2           # PPO clip bounds; only used when clip_ratio=True (also drives clipfrac diag)
    clip_coef_high: float = 0.28     # clip-higher (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5       # grad-norm clip per optimizer
    target_kl: Optional[float] = None  # NO KL early-stop

    # Backbone sharing. True => ONE ThinkTrunk for both heads via a single coherent
    # optimizer; False => separate actor/critic trunks with their own optimizers.
    # Default False to match the sepbb sibling run.
    share_backbone: bool = False

    # Dreamer3-bucket HL-Gauss distributional critic (ported from iterthink_v166).
    # The support lives in SYMLOG-COORDINATE space; raw bucket centers are
    # symexp(linspace(v_min, v_max, num_bins)) -> exponential Dreamer3 spacing.
    critic_num_bins: int = 511                  # odd -> one exact zero bucket
    critic_v_min: float = -9.90353755128617     # symlog-coord min; symexp ≈ -19999 raw
    critic_v_max: float = 9.90353755128617      # symlog-coord max; symexp ≈ +19999 raw
    critic_sigma_ratio: float = 0.75            # HL-Gauss projection sigma as a fraction of the coord bin width

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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
        # NO reward normalization: the critic regresses RAW returns (Dreamer3-bucket
        # symlog support absorbs the large, non-stationary scale).
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    # Plain pre-act MLP. Standard sqrt(2) init on both Linears.
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts

        self.in_proj = layer_init(nn.Linear(in_dim, H))
        # Bounded convex residual gate: g = sigmoid(resid_gate) per channel.
        # init +4 → g ≈ 0.982 → x_in ≈ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel γ).
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1) # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            block_in_dim = H * (k + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    """Separate (or optionally shared) ThinkTrunks: unimodal-Beta actor + Dreamer3-bucket
    HL-Gauss distributional critic. Actor and critic are queried independently so the two
    decoupled update loops never run the other head's trunk."""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # HL-Gauss distributional value head: num_bins logits over the Dreamer3-bucket support.
        # Plain init (std=1.0): the target is regressed scalar returns (no distributional
        # bootstrap), so the peaked-at-0 C51 init is unnecessary.
        self.critic_head = layer_init(nn.Linear(H, args.critic_num_bins), std=1.0)
        # Unimodal Beta heads: alpha,beta = 1 + softplus(.) > 1.
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )

    def _actor_feat(self, x):
        return self.trunk(x) if self.share_backbone else self.actor_trunk(x)

    def _critic_feat(self, x):
        return self.trunk(x) if self.share_backbone else self.critic_trunk(x)

    def _dist(self, actor_feat):
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        return Beta(alpha, beta)

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller decodes via the HL-Gauss support.
        return self.critic_head(self._critic_feat(x))

    def get_action(self, x, z=None):
        # Actor-only forward (no critic trunk). z is the NATIVE Beta sample in (0,1);
        # replayed from the buffer so log_prob is recomputed at the same draw.
        dist = self._dist(self._actor_feat(x))
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        log_prob = dist.log_prob(z).sum(1)            # constant rescale Jacobian drops out
        entropy = dist.entropy().sum(1)
        return z, action, log_prob, entropy


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

    # Dreamer3-bucket HL-Gauss categorical support for the distributional value critic.
    # symlog-coordinate buckets, symexp-spaced raw centers (exponential Dreamer3 spacing).
    hlg = Dreamer3BucketHLGaussSupport(
        args.critic_num_bins,
        args.critic_v_min,
        args.critic_v_max,
        args.critic_sigma_ratio,
        device,
    )

    agent = Agent(envs, args).to(device)
    # The actor (1 epoch) and critic (10 epochs) train in separate loops, each clipping
    # only its own param group. heads list which params receive each loss's gradient.
    actor_heads = list(agent.actor_alpha_head.parameters()) + list(agent.actor_beta_head.parameters())
    critic_heads = list(agent.critic_head.parameters())
    if args.share_backbone:
        # ONE shared trunk. A SINGLE optimizer over all params keeps the trunk's Adam
        # moments coherent across both loops (each step() only updates params that have a
        # grad: critic loop -> trunk+critic_head, actor loop -> trunk+actor_heads).
        trunk_params = list(agent.trunk.parameters())
        actor_params = trunk_params + actor_heads
        critic_params = trunk_params + critic_heads
        actor_opt = critic_opt = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        # Separate trunks -> two independent optimizers, fully decoupled.
        actor_params = list(agent.actor_trunk.parameters()) + actor_heads
        critic_params = list(agent.critic_trunk.parameters()) + critic_heads
        actor_opt = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
        critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    # Storage: store the NATIVE beta sample z (replayed to recompute logp at the same draw).
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
            actor_opt.param_groups[0]["lr"] = lrnow
            critic_opt.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                z, action, logprob, _ = agent.get_action(next_obs)
                values[step] = hlg.to_scalar(agent.get_value(next_obs)).flatten()
            zs[step] = z
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

        # GAE on the decoded scalar values -- U = RAW advantage (no normalization).
        with torch.no_grad():
            next_value = hlg.to_scalar(agent.get_value(next_obs)).reshape(1, -1)
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
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Dreamer3-bucket HL-Gauss target probs for the scalar returns (Gaussian-CDF mass
        # over the symlog-coordinate intervals, clamped to the coord support).
        b_target_probs = hlg.project(b_returns)

        b_inds = np.arange(args.batch_size)

        # ---- Critic update: many epochs of distributional regression (no policy / off-policy bias) ----
        critic_gns = []
        for epoch in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                value_logits = agent.get_value(b_obs[mb_inds])
                v_loss = -(b_target_probs[mb_inds] * F.log_softmax(value_logits, dim=-1)).sum(-1).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                critic_gns.append(float(nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)))
                critic_opt.step()

        # ---- Actor update: ONE unclipped (or clipped) PG pass on raw advantages (z-replay) ----
        actor_gns, clipfracs = [], []
        approx_kl = torch.zeros((), device=device)
        entropy_val = torch.zeros((), device=device)
        pg_loss = torch.zeros((), device=device)
        for epoch in range(args.actor_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_zs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                if args.clip_ratio:
                    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:
                    # NO ratio clip: plain importance-weighted PG surrogate.
                    pg_loss = (-mb_adv * ratio).mean()
                entropy_val = entropy.mean()
                actor_loss = pg_loss - args.ent_coef * entropy_val

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_gns.append(float(nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)))
                actor_opt.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_val.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(np.mean(actor_gns)), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(np.mean(critic_gns)), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        # With reward-norm OFF these are RAW returns/advantages (large, non-stationary);
        # the Dreamer3-bucket symlog critic is what keeps the value calibrated.
        writer.add_scalar("debug/raw_adv_rms", b_advantages.pow(2).mean().sqrt().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
