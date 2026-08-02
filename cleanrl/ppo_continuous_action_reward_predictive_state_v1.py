# PPO + low-rank future-reward predictive-state critic, v1 (RNG-paired).
#
# Hypothesis: scalar TD aliases distinct temporal causes and propagates reward only one
# Bellman edge per target refresh. The critic therefore predicts the identifiable raw
# future-reward sequence with a learned rank-16 factorization. Value is its discounted
# 128-step prefix plus a learned tail. Direct future rewards and one-step shift
# consistency train the predictive state; an exact scalar TD0 loss trains the composed
# value. PPO alone consumes true environment-reward GAE. Reward normalization is off.
import os
import random
import time
from dataclasses import dataclass
from math import log
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


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
    update_epochs: int = 10
    # ppoadvnorm_batch_v1 config, baked in: plain advantage standardization, once over
    # the whole rollout (batch scope), as the SOLE advantage treatment.
    norm_adv: bool = True
    norm_adv_scope: str = "batch"    # "batch" (reference) | "minibatch" (idiomatic PPO)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    vf_coef: float = 0.5             # weight on the explicitly balanced SF critic loss
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and SF heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)
    predictive_grad_clip: float = 0.25

    predictive_rank: int = 16
    prediction_horizon: int = 128
    future_reward_loss_coef: float = 1.0
    shift_loss_coef: float = 0.25
    tail_bridge_loss_coef: float = 0.25
    mc_window: int = 500             # gamma^500 ~= .0066: independent truncated-MC EV gate

    # Compile only the pure trunk. CUDA graphs remain disabled because the same compiled
    # trunk participates in separate critic and actor graphs inside each optimizer step.
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0
    logvar_max: float = 8.0

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class RecordRawObservation(gym.Wrapper):
    """Expose the flattened pre-normalization observation without changing it."""

    info_key = "raw_observation"

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info[self.info_key] = np.array(observation, copy=True)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[self.info_key] = np.array(observation, copy=True)
        return observation, reward, terminated, truncated, info


def raw_observations_from_infos(infos):
    """Stack the always-present per-env observations emitted through vector infos."""
    key = RecordRawObservation.info_key
    if key not in infos or not np.asarray(infos.get(f"_{key}", False)).all():
        raise RuntimeError("raw observation wrapper did not populate every vector info")
    return np.stack(infos[key]).astype(np.float32, copy=False)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = RecordRawObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def target_valid_mask(terminations, valids):
    return terminations.bool() | valids.bool()


def scalar_td0_target(rewards, next_values, terminations, valids, gamma):
    target_valid = target_valid_mask(terminations, valids)
    bootstrap = (1.0 - terminations) * valids
    target = rewards + gamma * bootstrap * next_values
    return torch.where(target_valid, target, torch.zeros_like(target)), target_valid


def scalar_gae(
    rewards, values, next_values, terminations, boundaries, valids, gamma, gae_lambda
):
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        target_valid = target_valid_mask(terminations[t], valids[t])
        bootstrap = (1.0 - terminations[t]) * valids[t]
        continuation = (1.0 - boundaries[t]) * target_valid
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        last = delta * target_valid + gamma * gae_lambda * continuation * last
        advantages[t] = last
    return advantages, values + advantages


def build_future_reward_targets(rewards, boundaries, horizon):
    """Observed reward vectors without crossing episodes or the rollout tail."""
    time_steps, num_envs = rewards.shape
    targets = rewards.new_zeros((time_steps, num_envs, horizon))
    valid = torch.zeros_like(targets, dtype=torch.bool)
    boundary_prefix = torch.cat(
        [torch.zeros_like(boundaries[:1]), boundaries.cumsum(dim=0)], dim=0
    )
    for offset in range(min(horizon, time_steps)):
        length = time_steps - offset
        targets[:length, :, offset] = rewards[offset:]
        crossed = boundary_prefix[offset : offset + length] - boundary_prefix[:length]
        valid[:length, :, offset] = crossed == 0
    return targets, valid


def masked_mse(prediction, target, valid):
    if not valid.any():
        return prediction.sum() * 0.0
    return (prediction[valid] - target[valid]).square().mean()


def normalize_valid_advantages(advantages, valid):
    normalized = torch.zeros_like(advantages)
    valid_advantages = advantages[valid]
    if valid_advantages.numel() == 0:
        return normalized
    normalized[valid] = (valid_advantages - valid_advantages.mean()) / (
        valid_advantages.std(unbiased=False) + 1e-8
    )
    return normalized


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


class CompiledTrunk(nn.Module):
    """Compile the tensor-only trunk without CUDA-graph buffer donation."""

    def __init__(self, trunk, mode):
        super().__init__()
        self.trunk = trunk
        if mode == "reduce-overhead":
            # reduce-overhead's principal extra is CUDA graphs, which are unsafe for
            # this algorithm's retained dual backward. Keep Inductor without graphs.
            self.compiled_forward = torch.compile(
                trunk.forward,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        else:
            self.compiled_forward = torch.compile(trunk.forward, mode=mode, dynamic=False)

    def forward(self, x):
        return self.compiled_forward(x)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)

        self.prediction_horizon = args.prediction_horizon
        self.predictive_rank = args.predictive_rank
        with torch.random.fork_rng(devices=[]):
            self.coefficient_head = layer_init(
                nn.Linear(H, self.predictive_rank), std=0.1
            )
            self.temporal_basis = layer_init(
                nn.Linear(self.predictive_rank, self.prediction_horizon), std=0.1
            )
            self.tail_head = layer_init(nn.Linear(H, 1), std=0.1)
            with torch.no_grad():
                self.coefficient_head.weight.zero_()
                self.coefficient_head.bias.zero_()
                self.tail_head.weight.zero_()
                self.tail_head.bias.zero_()

        # Consume the exact CPU RNG path of the baseline's bias-free 6x511 critic
        # head before actor heads are initialized. Its subsequent zero_ is RNG-free.
        baseline_rng_dummy = layer_init(nn.Linear(H, 6 * 511, bias=False), std=0.1)
        del baseline_rng_dummy
        self.register_buffer(
            "discount_weights",
            args.gamma ** torch.arange(self.prediction_horizon, dtype=torch.float32),
        )
        self.tail_discount = args.gamma**self.prediction_horizon

        # v24: action distribution (unchanged from base).
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn
        # beta
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_critic(self, x):
        _, critic_feat = self._trunks(x)
        coefficients = self.coefficient_head(critic_feat)
        future_rewards = self.temporal_basis(coefficients)
        tail = self.tail_head(critic_feat).squeeze(-1)
        value = future_rewards @ self.discount_weights + self.tail_discount * tail
        return coefficients, future_rewards, tail, value

    def get_value(self, x):
        return self.get_critic(x)[-1]

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        coefficients = self.coefficient_head(critic_feat)
        future_rewards = self.temporal_basis(coefficients)
        tail = self.tail_head(critic_feat).squeeze(-1)
        value = future_rewards @ self.discount_weights + self.tail_discount * tail
        if self.actor_dist == "gaussian":
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value, future_rewards

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return (
            list(trunk.parameters())
            + list(self.coefficient_head.parameters())
            + list(self.temporal_basis.parameters())
            + list(self.tail_head.parameters())
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert args.predictive_rank > 0
    assert args.prediction_horizon > 1
    assert args.mc_window > 0
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

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    if args.compile:
        # The trunk sees fixed tensor shapes and is the expensive pure-tensor path.
        # The distribution construction and retained dual backward remain eager.
        import torch._dynamo
        import torch._functorch.config

        torch._dynamo.config.suppress_errors = True
        torch._functorch.config.donated_buffer = False
        if agent.share_backbone:
            agent.trunk = CompiledTrunk(agent.trunk, args.compile_mode)
        else:
            agent.actor_trunk = CompiledTrunk(agent.actor_trunk, args.compile_mode)
            agent.critic_trunk = CompiledTrunk(agent.critic_trunk, args.compile_mode)
        print(
            f"torch.compile trunk mode={args.compile_mode!r}; "
            "CUDA graphs disabled for separate dual backward"
        )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_np, reset_infos = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
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
                action, z, logprob, ent, value, _ = (
                    agent.get_action_and_value(next_obs)
                )
                values[step] = value
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            flat_next_obses = next_obses.reshape((-1,) + envs.single_observation_space.shape)
            _, next_future_predictions, next_tails, next_transition_values = agent.get_critic(
                flat_next_obses
            )
            next_future_predictions = next_future_predictions.reshape(
                args.num_steps, args.num_envs, args.prediction_horizon
            )
            next_transition_values = next_transition_values.reshape(
                args.num_steps, args.num_envs
            )
            next_tails = next_tails.reshape(args.num_steps, args.num_envs)
            advantages, returns = scalar_gae(
                rewards,
                values,
                next_transition_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )
            td0_targets, td0_valid = scalar_td0_target(
                rewards,
                next_transition_values,
                transition_terminations,
                transition_valids,
                args.gamma,
            )
            reward_vector_targets, reward_vector_valid = build_future_reward_targets(
                rewards, transition_boundaries, args.prediction_horizon
            )
            shift_valid = (
                (1.0 - transition_terminations) * transition_valids
            ).bool()

            # Independent finite-episode Monte Carlo diagnostic. Rows in a rollout's
            # unfinished final episode are excluded; there is no value bootstrap.
            truncated_mc_returns = torch.zeros_like(rewards)
            truncated_mc_valid = torch.zeros_like(transition_boundaries, dtype=torch.bool)
            running_mc = torch.zeros(args.num_envs, device=device)
            available_mc = torch.zeros(args.num_envs, device=device)
            has_future_boundary = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            for t in reversed(range(args.num_steps)):
                boundary = transition_boundaries[t].bool()
                running_mc = rewards[t] + args.gamma * running_mc * (~boundary)
                available_mc = 1.0 + available_mc * (~boundary)
                has_future_boundary |= boundary
                truncated_mc_returns[t] = running_mc
                truncated_mc_valid[t] = has_future_boundary & (available_mc >= args.mc_window)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_td0_targets = td0_targets.reshape(-1)
        b_td0_valid = td0_valid.reshape(-1)
        b_actor_valid = b_td0_valid
        b_reward_vector_targets = reward_vector_targets.reshape(
            -1, args.prediction_horizon
        )
        b_reward_vector_valid = reward_vector_valid.reshape(
            -1, args.prediction_horizon
        )
        b_next_future_predictions = next_future_predictions.reshape(
            -1, args.prediction_horizon
        )
        b_next_tails = next_tails.reshape(-1)
        b_shift_valid = shift_valid.reshape(-1)
        b_truncated_mc_returns = truncated_mc_returns.reshape(-1)
        b_truncated_mc_valid = truncated_mc_valid.reshape(-1)
        # ppoadvnorm_batch: one z-score over the whole rollout.
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_adv_normed = normalize_valid_advantages(
                b_advantages, b_actor_valid
            )

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # Iteration-averaged loss accumulators (last-minibatch-only logging is
        # biased low and noisy — see red-team notes).
        acc = {
            "td0": 0.0,
            "future_reward": 0.0,
            "shift": 0.0,
            "tail_bridge": 0.0,
            "pg": 0.0,
            "actor_gn": 0.0,
            "critic_gn": 0.0,
            "predictive_gn": 0.0,
            "n": 0,
        }
        epochs_completed = 0
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        entropy_loss = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Critic forward is separate from the actor forward. Gradients are
                # still evaluated at identical parameters and summed before one Adam
                # step, but no graph is retained across two backward calls. Besides
                # reducing live activation memory, this makes torch.compile safe.
                _, _, _, new_value = agent.get_critic(b_obs[mb_inds])
                mb_td_valid = b_td0_valid[mb_inds]
                td0_loss = masked_mse(
                    new_value, b_td0_targets[mb_inds], mb_td_valid
                )
                scalar_value_loss = args.vf_coef * td0_loss

                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    scalar_value_loss.backward()
                    critic_gn = nn.utils.clip_grad_norm_(
                        critic_params, args.critic_grad_clip
                    )
                    value_grads = [
                        (parameter, parameter.grad.detach().clone())
                        for parameter in critic_params
                        if parameter.grad is not None
                    ]
                    optimizer.zero_grad(set_to_none=True)

                _, predicted_rewards, predicted_tail, _ = agent.get_critic(b_obs[mb_inds])
                future_reward_loss = masked_mse(
                    predicted_rewards,
                    b_reward_vector_targets[mb_inds],
                    b_reward_vector_valid[mb_inds],
                )
                mb_shift_valid = b_shift_valid[mb_inds]
                if mb_shift_valid.any():
                    shift_loss = F.mse_loss(
                        predicted_rewards[mb_shift_valid, 1:],
                        b_next_future_predictions[mb_inds][mb_shift_valid, :-1],
                    )
                else:
                    shift_loss = predicted_rewards.sum() * 0.0
                if mb_shift_valid.any():
                    tail_bridge_target = (
                        b_next_future_predictions[mb_inds][mb_shift_valid, -1]
                        + args.gamma * b_next_tails[mb_inds][mb_shift_valid]
                    )
                    tail_bridge_loss = F.mse_loss(
                        predicted_tail[mb_shift_valid], tail_bridge_target
                    )
                else:
                    tail_bridge_loss = predicted_tail.sum() * 0.0
                predictive_loss = (
                    args.future_reward_loss_coef * future_reward_loss
                    + args.shift_loss_coef * shift_loss
                    + args.tail_bridge_loss_coef * tail_bridge_loss
                )
                v_loss = scalar_value_loss + predictive_loss

                if args.separate_grad_clip:
                    predictive_loss.backward()
                    predictive_gn = nn.utils.clip_grad_norm_(
                        critic_params, args.predictive_grad_clip
                    )
                    predictive_grads = [
                        (parameter, parameter.grad.detach().clone())
                        for parameter in critic_params
                        if parameter.grad is not None
                    ]
                    optimizer.zero_grad(set_to_none=True)

                _, _, newlogprob, entropy, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                mb_actor_valid = b_actor_valid[mb_inds]
                if not mb_actor_valid.any():
                    continue
                logratio = (
                    newlogprob - b_logprobs[mb_inds]
                )[mb_actor_valid]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_adv_normed[mb_inds][mb_actor_valid]
                    else:
                        mb_advantages = normalize_valid_advantages(
                            b_advantages[mb_inds][mb_actor_valid],
                            torch.ones_like(
                                b_advantages[mb_inds][mb_actor_valid], dtype=torch.bool
                            ),
                        )
                else:
                    mb_advantages = b_advantages[mb_inds][mb_actor_valid]

                # Asymmetric "clip-higher": looser upper bound gives positive-advantage
                # actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                with torch.no_grad():
                    acc["td0"] += td0_loss.item()
                    acc["future_reward"] += future_reward_loss.item()
                    acc["shift"] += shift_loss.item()
                    acc["tail_bridge"] += tail_bridge_loss.item()
                    acc["pg"] += pg_loss.item()
                    acc["n"] += 1

                entropy_loss = entropy[mb_actor_valid].mean()

                if args.separate_grad_clip:
                    # Decoupled clipping: value and policy gradients come from separate
                    # forwards, are clipped independently, then summed on the shared
                    # trunk before the single optimizer step.
                    (pg_loss - args.ent_coef * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for parameter, gradient in value_grads + predictive_grads:
                        parameter.grad = (
                            gradient
                            if parameter.grad is None
                            else parameter.grad + gradient
                        )
                    optimizer.step()
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm
                    )
                    predictive_gn = critic_gn
                    optimizer.step()
                acc["critic_gn"] += float(critic_gn)
                acc["predictive_gn"] += float(predictive_gn)
                acc["actor_gn"] += float(actor_gn)

            epochs_completed = epoch + 1
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if b_truncated_mc_valid.any():
            mc_pred = b_values[b_truncated_mc_valid].cpu().numpy()
            mc_true = b_truncated_mc_returns[b_truncated_mc_valid].cpu().numpy()
            var_mc = np.var(mc_true)
            truncated_mc_ev = (
                np.nan if var_mc == 0 else 1 - np.var(mc_true - mc_pred) / var_mc
            )
        else:
            truncated_mc_ev = np.nan

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        n_mb = max(acc["n"], 1)
        writer.add_scalar(
            "losses/value_loss",
            (
                args.vf_coef * acc["td0"]
                + args.future_reward_loss_coef * acc["future_reward"]
                + args.shift_loss_coef * acc["shift"]
                + args.tail_bridge_loss_coef * acc["tail_bridge"]
            )
            / n_mb,
            global_step,
        )
        writer.add_scalar("losses/scalar_td0", acc["td0"] / n_mb, global_step)
        writer.add_scalar(
            "losses/future_reward", acc["future_reward"] / n_mb, global_step
        )
        writer.add_scalar("losses/shift", acc["shift"] / n_mb, global_step)
        writer.add_scalar(
            "losses/tail_bridge", acc["tail_bridge"] / n_mb, global_step
        )
        writer.add_scalar("losses/policy_loss", acc["pg"] / n_mb, global_step)
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar(
            "losses/clipfrac",
            np.mean(clipfracs) if clipfracs else np.nan,
            global_step,
        )
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "diagnostics/truncated_mc_explained_variance", truncated_mc_ev, global_step
        )
        writer.add_scalar("losses/actor_grad_norm", acc["actor_gn"] / n_mb, global_step)
        writer.add_scalar("losses/critic_grad_norm", acc["critic_gn"] / n_mb, global_step)
        writer.add_scalar(
            "losses/predictive_grad_norm", acc["predictive_gn"] / n_mb, global_step
        )
        with torch.no_grad():
            diagnostic_coefficients, diagnostic_future, diagnostic_tail, diagnostic_value = (
                agent.get_critic(b_obs)
            )
            prefix_value = diagnostic_future @ agent.discount_weights
            future_mse = masked_mse(
                diagnostic_future,
                b_reward_vector_targets,
                b_reward_vector_valid,
            )
            diagnostic_stride = max(diagnostic_future.shape[0] // 4096, 1)
            rank_sample = diagnostic_future[::diagnostic_stride]
            centered_future = rank_sample - rank_sample.mean(dim=0)
            singular_values = torch.linalg.svdvals(centered_future.float())
            singular_weights = singular_values.square()
            singular_weights = singular_weights / singular_weights.sum().clamp_min(1e-8)
            effective_rank = torch.exp(
                -(singular_weights * singular_weights.clamp_min(1e-8).log()).sum()
            )
            tail_contribution = agent.tail_discount * diagnostic_tail
            prefix_fraction = prefix_value.abs().mean() / (
                prefix_value.abs().mean() + tail_contribution.abs().mean()
            ).clamp_min(1e-8)
        writer.add_scalar("diagnostics/future_reward_mse", future_mse.item(), global_step)
        for bucket_start, bucket_end in ((0, 8), (8, 32), (32, 64), (64, 128)):
            bucket_end = min(bucket_end, args.prediction_horizon)
            if bucket_start >= bucket_end:
                continue
            bucket_mse = masked_mse(
                diagnostic_future[:, bucket_start:bucket_end],
                b_reward_vector_targets[:, bucket_start:bucket_end],
                b_reward_vector_valid[:, bucket_start:bucket_end],
            )
            writer.add_scalar(
                f"diagnostics/future_reward_mse_{bucket_start}_{bucket_end}",
                bucket_mse.item(),
                global_step,
            )
        writer.add_scalar(
            "diagnostics/future_reward_effective_rank", effective_rank.item(), global_step
        )
        writer.add_scalar(
            "diagnostics/prefix_value_fraction", prefix_fraction.item(), global_step
        )
        writer.add_scalar(
            "diagnostics/tail_abs_mean", tail_contribution.abs().mean().item(), global_step
        )
        writer.add_scalar(
            "diagnostics/coefficient_norm_mean",
            diagnostic_coefficients.norm(dim=-1).mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/temporal_basis_weight_norm",
            agent.temporal_basis.weight.norm().item(),
            global_step,
        )
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/td0_target_absmax", td0_targets.abs().max().item(), global_step)
        writer.add_scalar("debug/value_absmax", diagnostic_value.abs().max().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
