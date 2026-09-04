# PPO + IterThink v24 Beta + reward-anchored successor features (sffactor v4 RNG-paired).
# =====================================================================================
# v4 is an experimental-correctness fix, not a claimed algorithmic improvement. The
# real SF head is initialized without advancing the global torch RNG, then a discarded
# bias-free Linear(H, 6*511) consumes exactly the initialization path of the matched
# baseline critic before actor-head construction. Actor initialization and all later
# global torch randomness are therefore paired with the 511-bin, six-horizon baseline.
#
# HYPOTHESIS: successor features can help a PPO critic only if their auxiliary
# supervision cannot corrupt the value definition. We therefore make transition
# features exact and stationary in physical coordinates:
#
#   phi_t = [reward_t, raw_obs_t, action_t, action_t^2, 1]
#   V(s_t) = psi(s_t)[0]
#
# Coordinate zero is exactly the raw-reward lambda return; there is no learned reward
# probe w and no factorization residual. The remaining coordinates predict discounted
# raw transition statistics and act only as balanced auxiliary supervision. One h=0
# head replaces v2's shifted-future MTP heads: every output estimates successor
# features from the current state under the rollout policy.
#
# Raw reward is intentional and reward normalization is unavailable: a running reward
# scale would change the anchor's physical meaning and its targets over training. The
# policy still receives the base algorithm's running-normalized, clipped observations
# for comparability. A wrapper records observations immediately after flattening and
# before normalization; only the SF auxiliary coordinates see those raw observations.
# Thus the policy input normalizer may keep adapting while the SF output basis does not.
#
# Heads emit raw successor coordinates. A detached EMA of full-rollout centered target
# scale is used only to precondition per-coordinate errors; it never transforms
# predictions or targets. Exactly deterministic coordinates fall back to target RMS.
# The reward-coordinate loss and mean auxiliary loss receive explicit,
# dimension-independent weights.
#
# PPO/IterThink/Beta defaults, raw GAE, shared trunk, asymmetric clipping, and separate
# 0.25 actor/critic gradient clipping are unchanged. Known coupling remains: the actor
# and critic update the same trunk, and actor KL early stopping also limits the number
# of critic minibatch passes. Gradient norms and completed epochs make that visible.
#
# PRE-REGISTERED HalfCheetah seed-1 decision rule: PASS if return is >=6044 at 4M
# (90% of the 6716 baseline and above v1's 5799) and >=7450 at 8M (90% of the 8278
# baseline; v1 never reached 8M). KILL if it is at least 10% below v1 at both 2M and
# 4M, or if the exact-anchor invariant exceeds 1e-4/non-finite values occur. The
# conventional lambda-return EV is retained
# for historical comparison but is target-coupled; truncated-MC EV uses only complete
# rollout-contained episode suffixes with at least 500 reward terms, stops at
# boundaries, and is independent but higher variance and slightly finite-horizon biased.
# =====================================================================================
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

    # Reward-anchored successor features. The output basis stays raw; this running
    # scale is used only to precondition detached squared errors.
    sf_aux_coef: float = 1.0
    sf_target_scale_decay: float = 0.99
    sf_loss_eps: float = 1e-6
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

        # One raw-coordinate successor head. Coordinate zero is the value itself.
        self.raw_obs_dim = obs_dim
        self.act_dim = act_dim
        self.sf_dim = 1 + obs_dim + act_dim + act_dim + 1
        self.sf_block_slices = {
            "reward": slice(0, 1),
            "raw_obs": slice(1, 1 + obs_dim),
            "action": slice(1 + obs_dim, 1 + obs_dim + act_dim),
            "action2": slice(1 + obs_dim + act_dim, 1 + obs_dim + 2 * act_dim),
            "const": slice(self.sf_dim - 1, self.sf_dim),
        }
        # Do not let the differently shaped real head perturb the matched actor seed.
        with torch.random.fork_rng(devices=[]):
            self.psi_head = layer_init(nn.Linear(H, self.sf_dim, bias=True), std=0.1)
            with torch.no_grad():
                self.psi_head.weight.zero_()
                self.psi_head.bias.zero_()

        # Consume the exact CPU RNG path of the baseline's bias-free 6x511 critic
        # head before actor heads are initialized. Its subsequent zero_ is RNG-free.
        baseline_rng_dummy = layer_init(nn.Linear(H, 6 * 511, bias=False), std=0.1)
        del baseline_rng_dummy
        self.register_buffer("sf_target_scale", torch.ones(self.sf_dim))
        self.register_buffer("sf_target_scale_initialized", torch.tensor(False, dtype=torch.bool))

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

    def transition_features(self, reward, raw_obs, action):
        """Build phi_t in a fixed raw transition-coordinate system."""
        return torch.cat(
            [reward.unsqueeze(-1), raw_obs, action, action.square(), torch.ones_like(reward.unsqueeze(-1))],
            dim=-1,
        )

    @torch.no_grad()
    def update_sf_target_scale(self, targets, decay, eps):
        """Update centered loss scales once from all frozen rollout targets."""
        rollout_std = targets.float().std(dim=(0, 1), unbiased=False)
        rollout_rms = targets.float().square().mean(dim=(0, 1)).sqrt()
        # Constant features have zero centered spread but still need a finite loss
        # scale. All non-degenerate coordinates use centered spread so a large value
        # mean cannot drown the state-dependent deviations that make a baseline useful.
        rollout_scale = torch.where(rollout_std > eps, rollout_std, rollout_rms)
        # The final coordinate is intentionally constant and carries only a bias/mean;
        # centered scaling would amplify insignificant numerical variation.
        rollout_scale[-1] = rollout_rms[-1]
        if not torch.isfinite(rollout_scale).all():
            raise RuntimeError("non-finite successor-feature target scale")
        rollout_scale.clamp_min_(eps)
        if self.sf_target_scale_initialized.item():
            self.sf_target_scale.mul_(decay).add_(rollout_scale, alpha=1.0 - decay)
        else:
            self.sf_target_scale.copy_(rollout_scale)
            self.sf_target_scale_initialized.fill_(True)
        self.sf_target_scale.clamp_min_(eps)

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

    def get_psi(self, x):
        _, critic_feat = self._trunks(x)
        return self.psi_head(critic_feat)

    def get_value(self, x):
        return self.get_psi(x)[:, 0]

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
        psi = self.psi_head(critic_feat)
        if self.actor_dist == "gaussian":
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, psi

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
        return list(trunk.parameters()) + list(self.psi_head.parameters())


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert 0.0 <= args.sf_target_scale_decay < 1.0
    assert args.sf_loss_eps > 0.0
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
    psis = torch.zeros((args.num_steps, args.num_envs, agent.sf_dim)).to(device)
    raw_obses = torch.zeros((args.num_steps, args.num_envs, agent.raw_obs_dim)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_np, reset_infos = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
    next_raw_obs = torch.as_tensor(
        raw_observations_from_infos(reset_infos), device=device, dtype=torch.float32
    )
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            raw_obses[step] = next_raw_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, psi = agent.get_action_and_value(next_obs)
                psis[step] = psi
                values[step] = psi[:, 0]
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
            next_raw_obs = torch.as_tensor(
                raw_observations_from_infos(infos), device=device, dtype=torch.float32
            )
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            flat_next_obses = next_obses.reshape((-1,) + envs.single_observation_space.shape)
            next_psis = agent.get_psi(flat_next_obses).reshape(
                args.num_steps, args.num_envs, agent.sf_dim
            )
            next_transition_values = next_psis[..., 0]

            # SCALAR GAE — UNCHANGED PPO. Real env rewards, V = psi[..., 0].
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values

            # VECTOR SF-TD(lambda) targets Psi_t: exactly the scalar lambda-return
            # recursion with rewards -> phi(s_t) and values -> psi_old. Computed once
            # per rollout from the OLD psi (like PPO's `returns`), fixed across epochs.
            #   Psi_t = phi_t + gamma * [(1-lam) psi_old(s'_{t+1}) + lam Psi_{t+1}]
            # implemented via the GAE form: Psi_t = psi_old_t + A_t,
            #   A_t = delta_t + gamma*lam*(1-boundary)*A_{t+1},
            #   delta_t = phi_t + gamma*bnt_t*psi_old(s'_{t+1}) - psi_old_t.
            # At a truncation boundary the recursion falls back to the pure bootstrap
            # through the final observation (bnt=1 there); at termination bnt=0.
            phis = agent.transition_features(rewards, raw_obses, actions)
            psi_targets = torch.zeros_like(psis)
            last_vec_gaelam = torch.zeros(args.num_envs, agent.sf_dim, device=device)
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = ((1.0 - transition_terminations[t]) * transition_valids[t]).unsqueeze(-1)
                lambda_nonterminal = (1.0 - transition_boundaries[t]).unsqueeze(-1)
                vec_delta = phis[t] + args.gamma * next_psis[t] * bootstrap_nonterminal - psis[t]
                last_vec_gaelam = (
                    vec_delta
                    + args.gamma * args.gae_lambda * lambda_nonterminal * last_vec_gaelam
                )
                psi_targets[t] = psis[t] + last_vec_gaelam

            # Because phi[..., 0] is exactly the same raw reward used by scalar GAE,
            # these are algebraically identical up to floating-point roundoff.
            anchor_return_maxerr = (psi_targets[..., 0] - returns).abs().max()
            if not torch.isfinite(anchor_return_maxerr) or anchor_return_maxerr > 1e-4:
                raise RuntimeError(
                    f"reward-anchor invariant failed: max error={anchor_return_maxerr.item():.6g}"
                )
            agent.update_sf_target_scale(
                psi_targets, args.sf_target_scale_decay, args.sf_loss_eps
            )

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
        b_psi_targets = psi_targets.reshape(-1, agent.sf_dim)
        b_truncated_mc_returns = truncated_mc_returns.reshape(-1)
        b_truncated_mc_valid = truncated_mc_valid.reshape(-1)
        # ppoadvnorm_batch: one z-score over the whole rollout.
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_adv_normed = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # Iteration-averaged loss accumulators (last-minibatch-only logging is
        # biased low and noisy — see red-team notes).
        acc = {
            "direct_value": 0.0,
            "aux_sf": 0.0,
            "pg": 0.0,
            "actor_gn": 0.0,
            "critic_gn": 0.0,
            "n": 0,
        }
        acc.update({f"block_{name}": 0.0 for name in agent.sf_block_slices})
        epochs_completed = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Critic forward is separate from the actor forward. Gradients are
                # still evaluated at identical parameters and summed before one Adam
                # step, but no graph is retained across two backward calls. Besides
                # reducing live activation memory, this makes torch.compile safe.
                new_psi = agent.get_psi(b_obs[mb_inds])
                scaled_err = (
                    new_psi - b_psi_targets[mb_inds]
                ) / agent.sf_target_scale.detach()
                scaled_sqerr = scaled_err.square()
                direct_value_loss = scaled_sqerr[:, 0].mean()
                aux_sf_loss = scaled_sqerr[:, 1:].mean()
                v_loss = args.vf_coef * (
                    direct_value_loss + args.sf_aux_coef * aux_sf_loss
                )

                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    v_loss.backward()
                    critic_gn = nn.utils.clip_grad_norm_(
                        critic_params, args.critic_grad_clip
                    )
                    value_grads = [
                        (p, p.grad.detach().clone())
                        for p in critic_params
                        if p.grad is not None
                    ]
                    optimizer.zero_grad(set_to_none=True)

                _, _, newlogprob, entropy, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_adv_normed[mb_inds]
                    else:
                        mb_advantages = b_advantages[mb_inds]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                else:
                    mb_advantages = b_advantages[mb_inds]

                # Asymmetric "clip-higher": looser upper bound gives positive-advantage
                # actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                with torch.no_grad():
                    acc["direct_value"] += direct_value_loss.item()
                    acc["aux_sf"] += aux_sf_loss.item()
                    for name, block_slice in agent.sf_block_slices.items():
                        acc[f"block_{name}"] += scaled_sqerr[:, block_slice].mean().item()
                    acc["pg"] += pg_loss.item()
                    acc["n"] += 1

                entropy_loss = entropy.mean()

                if args.separate_grad_clip:
                    # Decoupled clipping: value and policy gradients come from separate
                    # forwards, are clipped independently, then summed on the shared
                    # trunk before the single optimizer step.
                    (pg_loss - args.ent_coef * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                acc["critic_gn"] += float(critic_gn)
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
            args.vf_coef
            * (acc["direct_value"] + args.sf_aux_coef * acc["aux_sf"])
            / n_mb,
            global_step,
        )
        writer.add_scalar("losses/direct_value_loss", acc["direct_value"] / n_mb, global_step)
        writer.add_scalar("losses/aux_sf_loss", acc["aux_sf"] / n_mb, global_step)
        writer.add_scalar("losses/policy_loss", acc["pg"] / n_mb, global_step)
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "diagnostics/truncated_mc_explained_variance", truncated_mc_ev, global_step
        )
        writer.add_scalar(
            "diagnostics/anchor_return_maxerr", anchor_return_maxerr.item(), global_step
        )
        writer.add_scalar("losses/actor_grad_norm", acc["actor_gn"] / n_mb, global_step)
        writer.add_scalar("losses/critic_grad_norm", acc["critic_gn"] / n_mb, global_step)
        for name, block_slice in agent.sf_block_slices.items():
            writer.add_scalar(
                f"sf_target_scale/{name}",
                agent.sf_target_scale[block_slice].mean().item(),
                global_step,
            )
            writer.add_scalar(
                f"sf_scaled_mse/{name}", acc[f"block_{name}"] / n_mb, global_step
            )
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/psi_target_absmax", psi_targets.abs().max().item(), global_step)
        writer.add_scalar("debug/psi_norm_mean", psis.norm(dim=-1).mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
