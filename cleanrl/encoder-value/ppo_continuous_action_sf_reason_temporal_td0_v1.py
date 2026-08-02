# PPO + reason- and timescale-resolved successor features (TD0), v1.
#
# Reward is exactly decomposed into task, control, and survival reasons, then
# lifted across several exponential horizons. One-step vector TD predicts every
# coordinate without learned-feature collapse. The task value is only the fixed sum readout of
# the requested-discount reason coordinates, so opposing returns cannot hide
# from either the TD0 or componentwise lambda-return learning signal.
#
# Q uses the full orthonormal degree-2 tensor-product Jacobi action field under
# the old factorized Beta policy. Its expectation is exactly zero, hence
# E_pi_old[psi_q(s,a)] = psi_v(s) by construction.  PPO continues to use scalar
# GAE while Q-V is inaccurate, then automatically replaces its noisy component
# using out-of-sample correlation-gated shrinkage. Observation normalization is
# only a trunk input transform; the successor basis uses physical reward units.
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
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter


SAMPLE_EPS = 1e-6


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
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    norm_adv_scope: str = "batch"
    clip_coef: float = 0.2
    clip_coef_high: float = 0.28
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    share_backbone: bool = True
    separate_grad_clip: bool = True
    actor_grad_clip: float = 0.25
    critic_grad_clip: float = 0.25

    # Exact reason values at the task discount get unit weight. Shorter-discount
    # predictions are cheap temporal auxiliaries and never enter the value readout.
    sf_return_coef: float = 1.0
    sf_aux_coef: float = 0.25
    sf_action_coef: float = 0.5
    sf_temporal_discounts: tuple[float, ...] = (0.0, 0.5, 0.9, 0.97)
    qadv_max_blend: float = 0.9
    qadv_slope_max: float = 2.0
    sf_target_scale_decay: float = 0.99
    sf_loss_eps: float = 1e-6
    mc_window: int = 500

    compile: bool = False
    compile_mode: str = "reduce-overhead"
    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class RecordRewardComponents(gym.Wrapper):
    """Expose an exact, common reward basis for the three benchmark tasks."""

    info_key = "reward_components"

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        base = self.env.unwrapped
        control = float(info.get("reward_ctrl", -base.control_cost(action)))
        survival = float(getattr(base, "healthy_reward", 0.0))
        task = float(info.get("reward_run", reward - control - survival))
        components = np.asarray([task, control, survival], dtype=np.float32)
        if not np.isclose(components.sum(dtype=np.float64), reward, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                f"reward decomposition mismatch: components={components}, reward={reward}"
            )
        info[self.info_key] = components
        return observation, reward, terminated, truncated, info


def reward_components_from_infos(infos, num_envs):
    """Recover transition components, including rows moved into final_info on autoreset."""
    key = RecordRewardComponents.info_key
    values = infos.get(key, [None] * num_envs)
    value_mask = np.asarray(infos.get(f"_{key}", key in infos), dtype=bool)
    if value_mask.ndim == 0:
        value_mask = np.full(num_envs, value_mask.item(), dtype=bool)
    final_infos = infos.get("final_info", [None] * num_envs)
    final_mask = np.asarray(infos.get("_final_info", "final_info" in infos), dtype=bool)
    if final_mask.ndim == 0:
        final_mask = np.full(num_envs, final_mask.item(), dtype=bool)
    components = []
    for env_idx in range(num_envs):
        if value_mask[env_idx] and values[env_idx] is not None:
            component = values[env_idx]
        elif (
            final_mask[env_idx]
            and final_infos[env_idx] is not None
            and key in final_infos[env_idx]
        ):
            component = final_infos[env_idx][key]
        else:
            raise RuntimeError(f"missing reward components for environment {env_idx}")
        components.append(component)
    return np.stack(components).astype(np.float32, copy=False)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = RecordRewardComponents(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # Deliberately no NormalizeReward: coordinate zero is the physical reward.
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if getattr(layer, "bias", None) is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


def _branch_body(hidden):
    return nn.Sequential(
        layer_init(nn.Linear(hidden, hidden)),
        ReLUSquared(),
        layer_init(nn.Linear(hidden, hidden)),
    )


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, hidden, n_experts):
        super().__init__()
        self.in_proj = layer_init(nn.Linear(in_dim, hidden))
        self.resid_gate = nn.Parameter(torch.full((hidden,), 4.0))
        self.dense_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.dense = _branch_body(hidden)
        self.moe_norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(hidden, n_experts))
        self.experts = nn.ModuleList([_branch_body(hidden) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        gate = torch.sigmoid(self.resid_gate)
        x_in = gate * x + (1.0 - gate) * x0
        dense_delta = self.dense(self.dense_norm(x_in))
        moe_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(moe_in), dim=-1)
        expert_outputs = torch.stack([expert(moe_in) for expert in self.experts], dim=1)
        moe_delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return x_in + dense_delta + moe_delta


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, hidden, n_blocks, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, hidden))
        self.blocks = nn.ModuleList(
            [ThinkBlock(hidden * (k + 1), hidden, n_experts) for k in range(n_blocks)]
        )
        cat_dim = hidden * (n_blocks + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, hidden))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class CompiledTrunk(nn.Module):
    def __init__(self, trunk, mode):
        super().__init__()
        self.trunk = trunk
        if mode == "reduce-overhead":
            self.compiled_forward = torch.compile(
                trunk.forward, dynamic=False, options={"triton.cudagraphs": False}
            )
        else:
            self.compiled_forward = torch.compile(trunk.forward, mode=mode, dynamic=False)

    def forward(self, x):
        return self.compiled_forward(x)


def correlation(x, y, eps=1e-8):
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(eps)


def compute_scalar_gae(
    rewards,
    values,
    next_values,
    terminations,
    boundaries,
    bootstrap_valids,
    gamma,
    gae_lambda,
):
    """GAE with Gymnasium termination, truncation, and missing-final-observation semantics."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        target_valid = torch.logical_or(terminations[t].bool(), bootstrap_valids[t].bool())
        bootstrap = (1.0 - terminations[t]) * bootstrap_valids[t]
        continuation = 1.0 - boundaries[t]
        while target_valid.ndim < rewards[t].ndim:
            target_valid = target_valid.unsqueeze(-1)
            bootstrap = bootstrap.unsqueeze(-1)
            continuation = continuation.unsqueeze(-1)
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        candidate = delta + gamma * gae_lambda * continuation * lastgaelam
        # A time-limit transition without final_observation has no valid target.
        # It is neither a terminal transition nor safe to connect across the reset.
        lastgaelam = torch.where(target_valid, candidate, torch.zeros_like(candidate))
        advantages[t] = lastgaelam
    return advantages, advantages + values


def masked_mean(x, mask):
    """Mean over valid rows while preserving any trailing feature dimensions."""
    if x.ndim == 1:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)
    expanded_mask = mask.view(mask.shape + (1,) * (x.ndim - mask.ndim))
    return (x * expanded_mask).sum() / (mask.sum().clamp_min(1.0) * x[0].numel())


def cancellation_aware_tolerance(terms, atol=1e-5, rtol=1e-5):
    """Floating-point identity tolerance that remains valid under opposing terms."""
    return atol + rtol * terms.abs().sum(dim=-1)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        hidden = args.hidden
        self.act_dim = act_dim
        self.reward_components = 3
        if any(discount < 0.0 or discount >= args.gamma for discount in args.sf_temporal_discounts):
            raise ValueError("auxiliary temporal discounts must satisfy 0 <= discount < gamma")
        self.temporal_discounts = (*args.sf_temporal_discounts, args.gamma)
        self.reward_feature_dim = self.reward_components * len(self.temporal_discounts)
        self.main_reward_start = self.reward_feature_dim - self.reward_components
        self.main_reward_slice = slice(self.main_reward_start, self.reward_feature_dim)
        self.action_basis_dim = 2 * act_dim + act_dim * (act_dim - 1) // 2
        self.sf_dim = self.reward_feature_dim
        self.share_backbone = args.share_backbone

        if self.share_backbone:
            self.trunk = ThinkTrunk(obs_dim, hidden, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, hidden, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, hidden, args.k_blocks, args.n_experts)

        # Critic initialization is RNG-isolated, then the baseline critic's RNG path
        # is consumed so actor initialization remains paired with the reference PPO.
        with torch.random.fork_rng(devices=[]):
            self.psi_v_head = layer_init(nn.Linear(hidden, self.sf_dim), std=0.1)
            self.psi_action_head = layer_init(
                nn.Linear(hidden, self.action_basis_dim * self.sf_dim), std=0.01
            )
            with torch.no_grad():
                self.psi_v_head.weight.zero_()
                self.psi_v_head.bias.zero_()
                self.psi_action_head.weight.zero_()
                self.psi_action_head.bias.zero_()
        baseline_rng_dummy = layer_init(nn.Linear(hidden, 6 * 511, bias=False), std=0.1)
        del baseline_rng_dummy

        self.actor_alpha_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

        sf_discounts = torch.tensor(
            [
                discount
                for discount in self.temporal_discounts
                for _ in range(self.reward_components)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("sf_discounts", sf_discounts)
        self.register_buffer("sf_target_scale", torch.ones(self.sf_dim))
        self.register_buffer("sf_target_scale_initialized", torch.tensor(False, dtype=torch.bool))

    def transition_features(self, reward_components):
        if reward_components.shape[-1] != self.reward_components:
            raise ValueError(f"expected {self.reward_components} reward components")
        reward_features = reward_components.unsqueeze(-2).expand(
            *reward_components.shape[:-1], len(self.temporal_discounts), self.reward_components
        ).reshape(*reward_components.shape[:-1], self.reward_feature_dim)
        return reward_features

    def scalar_value(self, psi):
        return psi[..., self.main_reward_slice].sum(dim=-1)

    @torch.no_grad()
    def update_sf_target_scale(self, targets, decay, eps):
        rollout_std = targets.float().std(dim=0, unbiased=False)
        rollout_rms = targets.float().square().mean(dim=0).sqrt()
        rollout_scale = torch.where(rollout_std > eps, rollout_std, rollout_rms)
        if not torch.isfinite(rollout_scale).all():
            raise RuntimeError("non-finite successor target scale")
        rollout_scale.clamp_min_(eps)
        if self.sf_target_scale_initialized.item():
            self.sf_target_scale.mul_(decay).add_(rollout_scale, alpha=1.0 - decay)
        else:
            self.sf_target_scale.copy_(rollout_scale)
            self.sf_target_scale_initialized.fill_(True)
        self.sf_target_scale.clamp_min_(eps)

    def _trunks(self, x):
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def _beta(self, actor_feat):
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        return Beta(alpha, beta)

    @staticmethod
    def _orthogonal_beta_basis(z, alpha, beta):
        """Unit-variance degree-1/2 Jacobi basis under Beta(alpha, beta).

        The quadratic is Gram-Schmidt orthogonalized against both the constant
        and linear functions.  Therefore every returned coordinate has exactly
        zero expectation under the stored old policy, including asymmetric Betas.
        """
        # Express the quadratic in standardized coordinates. This avoids the
        # cancellation between fourth central moments that occurs for sharply
        # concentrated policies when evaluated directly in float32.
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total.square() * (total + 1.0))
        centered = z - mean
        linear = centered / variance.sqrt().clamp_min(1e-8)
        skewness = (
            2.0
            * (beta - alpha)
            * (total + 1.0).sqrt()
            / ((total + 2.0) * (alpha * beta).sqrt())
        )
        excess_kurtosis = 6.0 * (
            (alpha - beta).square() * (total + 1.0)
            - alpha * beta * (total + 2.0)
        ) / (alpha * beta * (total + 2.0) * (total + 3.0))
        quadratic = linear.square() - 1.0 - skewness * linear
        quadratic_variance = 2.0 + excess_kurtosis - skewness.square()
        quadratic = quadratic / quadratic_variance.clamp_min(1e-8).sqrt()
        cross_terms = [
            linear[:, i] * linear[:, j]
            for i in range(linear.shape[-1])
            for j in range(i + 1, linear.shape[-1])
        ]
        if cross_terms:
            return torch.cat([linear, quadratic, torch.stack(cross_terms, dim=-1)], dim=-1)
        return torch.cat([linear, quadratic], dim=-1)

    def _action_advantage_from_feat(self, critic_feat, basis):
        # The action loss trains its field, not the state-value representation.
        # psi_v's own TD0 + lambda-return objectives exclusively shape that trunk.
        coeff = self.psi_action_head(critic_feat.detach()).view(
            -1, self.action_basis_dim, self.sf_dim
        )
        return torch.einsum("bpk,bp->bk", coeff, basis)

    def _psi_from_feat(self, critic_feat, z=None, old_alpha=None, old_beta=None):
        psi_v = self.psi_v_head(critic_feat)
        if z is None:
            return psi_v, None, None
        if old_alpha is None or old_beta is None:
            raise ValueError("old Beta parameters are required for action-conditioned psi_q")
        basis = self._orthogonal_beta_basis(z, old_alpha, old_beta)
        action_advantage = self._action_advantage_from_feat(critic_feat, basis)
        return psi_v, action_advantage, basis

    def get_psi_v(self, x):
        _, critic_feat = self._trunks(x)
        return self.psi_v_head(critic_feat)

    def get_psi_v_action_advantage(self, x, basis):
        _, critic_feat = self._trunks(x)
        psi_v = self.psi_v_head(critic_feat)
        return psi_v, self._action_advantage_from_feat(critic_feat, basis)

    def get_value(self, x):
        return self.scalar_value(self.get_psi_v(x))

    def get_action_and_value(self, x, z=None, return_q=False):
        actor_feat, critic_feat = self._trunks(x)
        dist = self._beta(actor_feat)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        log_prob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        psi_v, action_advantage, action_basis = self._psi_from_feat(
            critic_feat,
            z if return_q else None,
            dist.concentration1 if return_q else None,
            dist.concentration0 if return_q else None,
        )
        return (
            action,
            z,
            log_prob,
            entropy,
            psi_v,
            action_advantage,
            action_basis,
        )

    def actor_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        return (
            list(trunk.parameters())
            + list(self.actor_alpha_head.parameters())
            + list(self.actor_beta_head.parameters())
        )

    def value_critic_parameters(self):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.psi_v_head.parameters())

    def action_critic_parameters(self):
        return list(self.psi_action_head.parameters())

    def critic_parameters(self):
        return self.value_critic_parameters() + self.action_critic_parameters()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert args.batch_size % args.num_minibatches == 0
    assert 0.0 <= args.sf_target_scale_decay < 1.0
    assert args.sf_loss_eps > 0.0
    assert args.mc_window > 0
    assert 0.0 <= args.qadv_max_blend <= 1.0
    assert args.qadv_slope_max > 0.0

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
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    agent = Agent(envs, args).to(device)
    if args.compile:
        import torch._dynamo
        import torch._functorch.config

        torch._dynamo.config.suppress_errors = True
        torch._functorch.config.donated_buffer = False
        if agent.share_backbone:
            agent.trunk = CompiledTrunk(agent.trunk, args.compile_mode)
        else:
            agent.actor_trunk = CompiledTrunk(agent.actor_trunk, args.compile_mode)
            agent.critic_trunk = CompiledTrunk(agent.critic_trunk, args.compile_mode)
        print(f"torch.compile trunk mode={args.compile_mode!r}; CUDA graphs disabled")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_params = agent.actor_parameters()
    value_critic_params = agent.value_critic_parameters()
    action_critic_params = agent.action_critic_parameters()
    critic_params = agent.critic_parameters()

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape, device=device)
    latent_zs = torch.zeros_like(actions)
    action_bases = torch.zeros(
        (args.num_steps, args.num_envs, agent.action_basis_dim), device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros_like(logprobs)
    reward_components = torch.zeros(
        (args.num_steps, args.num_envs, agent.reward_components), device=device
    )
    values = torch.zeros_like(logprobs)
    q_values = torch.zeros_like(logprobs)
    psi_vs = torch.zeros((args.num_steps, args.num_envs, agent.sf_dim), device=device)
    psi_action_advantages = torch.zeros_like(psi_vs)
    transition_terminations = torch.zeros_like(logprobs)
    transition_boundaries = torch.zeros_like(logprobs)
    transition_valids = torch.zeros_like(logprobs)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
    lagged_qadv_slope = torch.zeros((), device=device)
    lagged_qadv_blend = torch.zeros((), device=device)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z, logprob, _, psi_v, action_advantage, action_basis = (
                    agent.get_action_and_value(next_obs, return_q=True)
                )
                psi_vs[step] = psi_v
                psi_action_advantages[step] = action_advantage
                values[step] = agent.scalar_value(psi_v)
                q_values[step] = agent.scalar_value(psi_v + action_advantage)
            actions[step] = action
            latent_zs[step] = z
            action_bases[step] = action_basis
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [item is not None for item in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            reward_components[step] = torch.as_tensor(
                reward_components_from_infos(infos, args.num_envs),
                device=device,
                dtype=torch.float32,
            )
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(
                transition_valid, device=device, dtype=torch.float32
            )
            next_obses[step] = torch.as_tensor(
                transition_next_obs, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            flat_next_obses = next_obses.reshape((-1,) + obs_shape)
            next_psi_vs = agent.get_psi_v(flat_next_obses).reshape(
                args.num_steps, args.num_envs, agent.sf_dim
            )
            next_values = agent.scalar_value(next_psi_vs)

            advantages, returns = compute_scalar_gae(
                rewards,
                values,
                next_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )
            main_reason_values = psi_vs[..., agent.main_reward_slice]
            next_main_reason_values = next_psi_vs[..., agent.main_reward_slice]
            reason_advantages, reason_returns = compute_scalar_gae(
                reward_components,
                main_reason_values,
                next_main_reason_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )
            reason_return_sum_error = (reason_returns.sum(dim=-1) - returns).abs()
            reason_return_sum_maxerr = reason_return_sum_error.max()
            reason_return_tolerance = cancellation_aware_tolerance(
                reason_returns, atol=1e-4
            )
            if (reason_return_sum_error > reason_return_tolerance).any():
                raise RuntimeError(
                    "reason lambda returns do not sum to scalar return; "
                    f"max error={reason_return_sum_maxerr.item():.6g}"
                )

            phis = agent.transition_features(reward_components)
            bootstrap = ((1.0 - transition_terminations) * transition_valids).unsqueeze(-1)
            successor_valid = torch.logical_or(
                transition_terminations.bool(), transition_valids.bool()
            )
            psi_targets = phis + bootstrap * agent.sf_discounts * next_psi_vs
            scalar_reward_target = rewards + args.gamma * bootstrap[..., 0] * next_values
            reward_target_error = (
                agent.scalar_value(psi_targets) - scalar_reward_target
            ).abs()
            reward_target_maxerr = reward_target_error.max()
            reward_target_terms = reward_components.abs() + (
                args.gamma
                * bootstrap
                * next_psi_vs[..., agent.main_reward_slice].abs()
            )
            reward_target_tolerance = cancellation_aware_tolerance(reward_target_terms)
            if not torch.isfinite(psi_targets).all() or (
                reward_target_error > reward_target_tolerance
            ).any():
                raise RuntimeError(
                    f"invalid TD0 successor target; reward max error={reward_target_maxerr.item():.6g}"
                )
            agent.update_sf_target_scale(
                psi_targets[successor_valid], args.sf_target_scale_decay, args.sf_loss_eps
            )

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

        b_obs = obs.reshape((-1,) + obs_shape)
        b_latent_zs = latent_zs.reshape((-1,) + action_shape)
        b_action_bases = action_bases.reshape(-1, agent.action_basis_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_reason_returns = reason_returns.reshape(-1, agent.reward_components)
        b_values = values.reshape(-1)
        b_q_values = q_values.reshape(-1)
        b_psi_vs = psi_vs.reshape(-1, agent.sf_dim)
        b_action_advantages = psi_action_advantages.reshape(-1, agent.sf_dim)
        b_psi_targets = psi_targets.reshape(-1, agent.sf_dim)
        b_action_targets = b_psi_targets - b_psi_vs
        b_successor_valid = successor_valid.reshape(-1)
        b_truncated_mc_returns = truncated_mc_returns.reshape(-1)
        b_truncated_mc_valid = truncated_mc_valid.reshape(-1)
        # The action field was fitted only on earlier rollouts, so its agreement
        # with this rollout's GAE is an online out-of-sample reliability estimate.
        # Its calibration is delayed to the next rollout, making it fixed before
        # every action whose score it weights is sampled.
        with torch.no_grad():
            rollout_qadv = b_q_values - b_values
            valid_gae = b_advantages[b_successor_valid]
            valid_qadv = rollout_qadv[b_successor_valid]
            centered_gae = valid_gae - valid_gae.mean()
            centered_qadv = valid_qadv - valid_qadv.mean()
            qadv_covariance = (centered_gae * centered_qadv).mean()
            qadv_variance = centered_qadv.square().mean()
            next_qadv_slope = (qadv_covariance / qadv_variance.clamp_min(1e-8)).clamp(
                0.0, args.qadv_slope_max
            )
            qadv_reliability = correlation(valid_qadv, valid_gae).clamp(0.0, 1.0).square()
            next_qadv_blend = args.qadv_max_blend * qadv_reliability
            b_policy_advantages = b_advantages.clone()
            b_policy_advantages[b_successor_valid] = (
                (1.0 - lagged_qadv_blend) * valid_gae
                + lagged_qadv_blend * lagged_qadv_slope * valid_qadv
            )
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_adv_normed = torch.zeros_like(b_advantages)
            valid_advantages = b_policy_advantages[b_successor_valid]
            b_adv_normed[b_successor_valid] = (
                valid_advantages - valid_advantages.mean()
            ) / (valid_advantages.std() + 1e-8)

        indices = np.arange(args.batch_size)
        clipfracs = []
        reason_return_scale = (
            b_reason_returns[b_successor_valid]
            .std(dim=0, unbiased=False)
            .detach()
            .clamp_min(1.0)
        )
        acc = {"v_reward": 0.0, "return": 0.0, "action_reward": 0.0,
               "v_aux": 0.0, "action_aux": 0.0,
               "pg": 0.0, "actor_gn": 0.0, "critic_gn": 0.0,
               "action_critic_gn": 0.0, "n": 0}
        epochs_completed = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = indices[start : start + args.minibatch_size]
                new_v, new_action_advantage = agent.get_psi_v_action_advantage(
                    b_obs[mb], b_action_bases[mb]
                )
                mb_valid = b_successor_valid[mb].to(new_v.dtype)
                scale = agent.sf_target_scale.detach()
                v_sqerr = ((new_v - b_psi_targets[mb]) / scale).square()
                # Regress only the action field to the frozen TD innovation.  psi_v
                # is absent from this graph, so finite-sample action-basis imbalance
                # cannot pull the state value away from its conditional expectation.
                old_action_target = b_psi_targets[mb] - b_psi_vs[mb]
                action_sqerr = ((new_action_advantage - old_action_target) / scale).square()
                v_reward_loss = masked_mean(
                    v_sqerr[:, agent.main_reward_slice], mb_valid
                )
                return_loss = masked_mean(
                    (
                        (
                            new_v[:, agent.main_reward_slice]
                            - b_reason_returns[mb]
                        )
                        / reason_return_scale
                    ).square(),
                    mb_valid,
                )
                action_reward_loss = masked_mean(
                    action_sqerr[:, agent.main_reward_slice], mb_valid
                )
                v_aux_loss = masked_mean(
                    v_sqerr[:, : agent.main_reward_start], mb_valid
                )
                action_aux_loss = masked_mean(
                    action_sqerr[:, : agent.main_reward_start], mb_valid
                )
                state_successor_loss = (
                    v_reward_loss
                    + args.sf_return_coef * return_loss
                    + args.sf_aux_coef * v_aux_loss
                )
                action_successor_loss = args.sf_action_coef * (
                    action_reward_loss + args.sf_aux_coef * action_aux_loss
                )
                state_v_loss = args.vf_coef * state_successor_loss
                action_v_loss = args.vf_coef * action_successor_loss

                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    state_v_loss.backward()
                    critic_gn = nn.utils.clip_grad_norm_(
                        value_critic_params, args.critic_grad_clip
                    )
                    critic_grads = [
                        (p, p.grad.detach().clone())
                        for p in value_critic_params
                        if p.grad is not None
                    ]
                    optimizer.zero_grad(set_to_none=True)

                    action_v_loss.backward()
                    action_critic_gn = nn.utils.clip_grad_norm_(
                        action_critic_params, args.critic_grad_clip
                    )
                    critic_grads.extend(
                        (p, p.grad.detach().clone())
                        for p in action_critic_params
                        if p.grad is not None
                    )
                    optimizer.zero_grad(set_to_none=True)

                _, _, newlogprob, entropy, _, _, _ = agent.get_action_and_value(
                    b_obs[mb], b_latent_zs[mb], return_q=False
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_adv_normed[mb]
                    else:
                        mb_advantages = torch.zeros_like(b_advantages[mb])
                        valid_mb_advantages = b_policy_advantages[mb][mb_valid.bool()]
                        mb_advantages[mb_valid.bool()] = (
                            valid_mb_advantages - valid_mb_advantages.mean()
                        ) / (valid_mb_advantages.std() + 1e-8)
                else:
                    mb_advantages = b_policy_advantages[mb]
                clip_high = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                pg_loss = torch.maximum(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + clip_high
                    ),
                )
                pg_loss = masked_mean(pg_loss, mb_valid)
                entropy_loss = masked_mean(entropy, mb_valid)

                if args.separate_grad_clip:
                    (pg_loss - args.ent_coef * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for parameter, gradient in critic_grads:
                        parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    total_v_loss = state_v_loss + action_v_loss
                    (pg_loss - args.ent_coef * entropy_loss + total_v_loss).backward()
                    actor_gn = critic_gn = nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm
                    )
                    action_critic_gn = critic_gn
                    optimizer.step()

                acc["v_reward"] += v_reward_loss.item()
                acc["return"] += return_loss.item()
                acc["action_reward"] += action_reward_loss.item()
                acc["v_aux"] += v_aux_loss.item()
                acc["action_aux"] += action_aux_loss.item()
                acc["pg"] += pg_loss.item()
                acc["actor_gn"] += float(actor_gn)
                acc["critic_gn"] += float(critic_gn)
                acc["action_critic_gn"] += float(action_critic_gn)
                acc["n"] += 1

            epochs_completed = epoch + 1
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values[b_successor_valid].cpu().numpy()
        y_true = b_returns[b_successor_valid].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
        if b_truncated_mc_valid.any():
            mc_pred = b_values[b_truncated_mc_valid].cpu().numpy()
            mc_true = b_truncated_mc_returns[b_truncated_mc_valid].cpu().numpy()
            var_mc = np.var(mc_true)
            truncated_mc_ev = np.nan if var_mc == 0 else 1.0 - np.var(mc_true - mc_pred) / var_mc
        else:
            truncated_mc_ev = np.nan

        with torch.no_grad():
            q_minus_v = (b_q_values - b_values)[b_successor_valid]
            valid_advantages = b_advantages[b_successor_valid]
            qv_corr = correlation(q_minus_v, valid_advantages)
            qv_std_ratio = q_minus_v.std() / valid_advantages.std().clamp_min(1e-8)
            qv_ev = 1.0 - (valid_advantages - q_minus_v).var() / valid_advantages.var().clamp_min(1e-8)
            qv_sign_agree = (q_minus_v.sign() == valid_advantages.sign()).float().mean()
            valid_psi = b_psi_vs[b_successor_valid]
            valid_action_predictions = b_action_advantages[b_successor_valid]
            valid_action_targets = b_action_targets[b_successor_valid]
            scaled_action_error = (
                (valid_action_predictions - valid_action_targets) / agent.sf_target_scale
            )
            action_td_mse = scaled_action_error.square().mean()
            action_reward_corr = correlation(
                agent.scalar_value(valid_action_predictions),
                agent.scalar_value(valid_action_targets),
            )
            main_reason_values = valid_psi[:, agent.main_reward_slice]

        n_mb = max(acc["n"], 1)
        mean_successor_loss = (
            acc["v_reward"] + args.sf_return_coef * acc["return"]
            + args.sf_aux_coef * acc["v_aux"]
            + args.sf_action_coef
            * (acc["action_reward"] + args.sf_aux_coef * acc["action_aux"])
        ) / n_mb
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", args.vf_coef * mean_successor_loss, global_step)
        writer.add_scalar("losses/sf_v_reward", acc["v_reward"] / n_mb, global_step)
        writer.add_scalar("losses/sf_reason_lambda_return", acc["return"] / n_mb, global_step)
        writer.add_scalar("losses/sf_action_reward", acc["action_reward"] / n_mb, global_step)
        writer.add_scalar("losses/sf_v_temporal", acc["v_aux"] / n_mb, global_step)
        writer.add_scalar("losses/sf_action_temporal", acc["action_aux"] / n_mb, global_step)
        writer.add_scalar("losses/policy_loss", acc["pg"] / n_mb, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", acc["actor_gn"] / n_mb, global_step)
        writer.add_scalar("losses/critic_grad_norm", acc["critic_gn"] / n_mb, global_step)
        writer.add_scalar(
            "losses/action_critic_grad_norm", acc["action_critic_gn"] / n_mb, global_step
        )
        writer.add_scalar("diagnostics/truncated_mc_explained_variance", truncated_mc_ev, global_step)
        writer.add_scalar("diagnostics/qv_gae_corr", qv_corr.item(), global_step)
        writer.add_scalar("diagnostics/qv_gae_ev", qv_ev.item(), global_step)
        writer.add_scalar("diagnostics/qv_gae_std_ratio", qv_std_ratio.item(), global_step)
        writer.add_scalar("diagnostics/qv_gae_sign_agree", qv_sign_agree.item(), global_step)
        writer.add_scalar("diagnostics/qv_sample_mean", q_minus_v.mean().item(), global_step)
        writer.add_scalar("diagnostics/qadv_slope", lagged_qadv_slope.item(), global_step)
        writer.add_scalar("diagnostics/qadv_blend", lagged_qadv_blend.item(), global_step)
        writer.add_scalar("diagnostics/qadv_next_slope", next_qadv_slope.item(), global_step)
        writer.add_scalar("diagnostics/qadv_next_blend", next_qadv_blend.item(), global_step)
        writer.add_scalar("diagnostics/action_td_innovation_mse", action_td_mse.item(), global_step)
        writer.add_scalar("diagnostics/action_reward_target_corr", action_reward_corr.item(), global_step)
        main_reason_action = b_action_advantages[:, agent.main_reward_slice]
        for reason_idx, reason_name in enumerate(("task", "control", "survival")):
            writer.add_scalar(
                f"diagnostics/action_{reason_name}_absmean",
                main_reason_action[:, reason_idx].abs().mean().item(),
                global_step,
            )
        writer.add_scalar(
            "sf/target_scale_reward",
            agent.sf_target_scale[agent.main_reward_slice].mean().item(),
            global_step,
        )
        writer.add_scalar(
            "sf/target_scale_temporal",
            agent.sf_target_scale[: agent.main_reward_start].mean().item(),
            global_step,
        )
        for reason_idx, reason_name in enumerate(("task", "control", "survival")):
            writer.add_scalar(
                f"sf/value_{reason_name}_mean",
                main_reason_values[:, reason_idx].mean().item(),
                global_step,
            )
        writer.add_scalar("sf/reward_target_maxerr", reward_target_maxerr.item(), global_step)
        writer.add_scalar(
            "sf/reason_return_sum_maxerr", reason_return_sum_maxerr.item(), global_step
        )
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # Calibration is consumed only on the next rollout, so it is fixed before
        # any action whose score it weights is sampled.
        lagged_qadv_slope = next_qadv_slope
        lagged_qadv_blend = next_qadv_blend

    envs.close()
    writer.close()
