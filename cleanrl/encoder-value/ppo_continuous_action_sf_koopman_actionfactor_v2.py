# PPO + action-factorized reward-observable Koopman resolvent critic, v2.
#
# A small whitened latent x=[1,h(raw_obs)] is trained to be closed under the
# one-step policy dynamics and linearly decode the exact task/control/survival
# reward reasons. Full-rollout ridge fits the nonnormal Galerkin operator K and
# reward map B; G=B(I-gamma K)^-1 analytically propagates every reward-relevant
# mode to the infinite discounted horizon without hand-selected timescales.
#
# Two full degree-2 canonical-monomial fields separately predict immediate
# reward and next-latent innovations. Each raw monomial is centered by its exact
# expectation under the old Beta policy, without policy-dependent rescaling, so
# coefficient semantics remain invariant as the policy changes. Their composition
# A_reason=e_r(a)+gamma*G*e_x(a) exposes which part of action value comes from
# immediate reward versus changed future occupancy, while retaining exactly
# zero old-policy expectation. PPO calibrates the frozen field on the current
# rollout's held-out environment before use and keeps scalar GAE as fallback.
# Model inputs use symlog physical observations, while policy observations keep
# the reference running normalization. Rewards are never normalized.
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
    target_kl: float = 0.03

    actor_grad_clip: float = 0.25
    critic_grad_clip: float = 0.25

    model_dim: int = 12
    model_hidden: int = 64
    model_lr: float = 1e-4
    model_epochs: int = 4
    operator_ridge: float = 1e-3
    model_reward_coef: float = 1.0
    model_closure_coef: float = 1.0
    model_td_coef: float = 1.0
    model_whiten_coef: float = 0.1
    action_lr: float = 3e-4
    action_epochs: int = 4
    qadv_max_blend: float = 0.9
    qadv_slope_max: float = 2.0
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


class RecordRawObservation(gym.Wrapper):
    """Put the flattened pre-normalization observation in every reset/step info."""

    info_key = "raw_observation"

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info[self.info_key] = np.array(observation, dtype=np.float32, copy=True)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[self.info_key] = np.array(observation, dtype=np.float32, copy=True)
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


def raw_observations_from_infos(infos, num_envs, boundaries=None):
    """Return emitted raw observations and transition-final raw observations.

    Under autoreset, the top-level row is the reset observation and final_info
    owns the transition's actual successor. Missing final raw observations are
    represented by a false transition-valid mask so truncations can be censored.
    """
    key = RecordRawObservation.info_key
    values = infos.get(key, [None] * num_envs)
    value_mask = np.asarray(infos.get(f"_{key}", key in infos), dtype=bool)
    if value_mask.ndim == 0:
        value_mask = np.full(num_envs, value_mask.item(), dtype=bool)
    has_final_infos = "final_info" in infos
    final_infos = infos.get("final_info", [None] * num_envs)
    final_mask = np.asarray(infos.get("_final_info", "final_info" in infos), dtype=bool)
    if final_mask.ndim == 0:
        final_mask = np.full(num_envs, final_mask.item(), dtype=bool)
    if boundaries is None:
        boundaries = np.zeros(num_envs, dtype=bool)
    emitted, transition, transition_valid = [], [], []
    for env_idx in range(num_envs):
        top = values[env_idx] if value_mask[env_idx] else None
        final = None
        if final_mask[env_idx] and final_infos[env_idx] is not None:
            final = final_infos[env_idx].get(key)
        if top is None:
            # Some vector modes expose only final_info on terminal rows. The
            # caller cannot continue from that row safely, so fail explicitly.
            raise RuntimeError(f"missing emitted raw observation for environment {env_idx}")
        emitted.append(top)
        if boundaries[env_idx]:
            # NEXT_STEP autoreset exposes the terminal observation at top level;
            # SAME_STEP autoreset moves it into final_info and exposes reset state.
            terminal_raw = final if final is not None else (None if has_final_infos else top)
            transition.append(np.zeros_like(top) if terminal_raw is None else terminal_raw)
            transition_valid.append(terminal_raw is not None)
        else:
            transition.append(top)
            transition_valid.append(True)
    return (
        np.stack(emitted).astype(np.float32, copy=False),
        np.stack(transition).astype(np.float32, copy=False),
        np.asarray(transition_valid, dtype=bool),
    )


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = RecordRewardComponents(env)
        env = RecordRawObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # Deliberately no NormalizeReward: every reason remains in physical units.
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


def symlog(x):
    return x.sign() * torch.log1p(x.abs())


def whitening_loss(latent):
    """Batch mean/covariance penalty; latent excludes the fixed constant."""
    latent = latent.float()
    mean = latent.mean(dim=0)
    centered = latent - mean
    covariance = centered.T @ centered / max(latent.shape[0], 1)
    identity = torch.eye(latent.shape[-1], device=latent.device, dtype=latent.dtype)
    return mean.square().mean() + (covariance - identity).square().mean()


@torch.no_grad()
def project_l2_contractive(koopman, metric, max_singular_value=1.0):
    """Project K to a contraction in the empirical feature L2 metric."""
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    floor = torch.finfo(metric.dtype).eps * eigenvalues.max().clamp_min(1.0)
    eigenvalues = eigenvalues.clamp_min(floor)
    metric_sqrt = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    metric_inv_sqrt = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.T
    whitened = metric_inv_sqrt @ koopman @ metric_sqrt
    left, singular_values, right_t = torch.linalg.svd(whitened, full_matrices=False)
    contracted = (
        left * singular_values.clamp_max(max_singular_value).unsqueeze(0)
    ) @ right_t
    return metric_sqrt @ contracted @ metric_inv_sqrt


@torch.no_grad()
def fit_koopman_resolvent(x, y, rewards, gamma, ridge):
    """Fit y=Kx, r=Bx and G=B(I-gamma K)^-1 in column orientation."""
    if x.ndim != 2 or y.shape != x.shape or rewards.ndim != 2:
        raise ValueError("invalid operator-fit shapes")
    if x.shape[0] != rewards.shape[0] or x.shape[0] == 0:
        raise ValueError("operator fit needs aligned nonempty rows")
    dtype = torch.float64
    xd, yd, rd = x.to(dtype), y.to(dtype), rewards.to(dtype)
    count = float(x.shape[0])
    cxx = xd.T @ xd / count
    regularizer = torch.eye(x.shape[1], device=x.device, dtype=dtype) * ridge
    regularizer[0, 0] = 0.0  # do not shrink the intercept/continuation coordinate
    gram = cxx + regularizer
    cyx = yd.T @ xd / count
    crx = rd.T @ xd / count
    # Right multiplication by Cxx^-1, without materializing an inverse.
    koopman = torch.linalg.solve(gram.T, cyx.T).T
    reward_map = torch.linalg.solve(gram.T, crx.T).T
    # Conditional expectation is nonexpansive in L2. Project in the empirical
    # feature metric, not the arbitrary coordinate metric; this guarantees the
    # discounted Neumann series while retaining a full nonnormal real operator.
    koopman = project_l2_contractive(koopman, gram)
    resolvent_matrix = torch.eye(x.shape[1], device=x.device, dtype=dtype) - gamma * koopman
    value_map = torch.linalg.solve(resolvent_matrix.T, reward_map.T).T
    condition = torch.linalg.cond(resolvent_matrix)
    if not (
        torch.isfinite(koopman).all()
        and torch.isfinite(reward_map).all()
        and torch.isfinite(value_map).all()
        and torch.isfinite(condition)
    ):
        raise RuntimeError("non-finite Koopman resolvent fit")
    return (
        koopman.float(),
        reward_map.float(),
        value_map.float(),
        condition.float(),
        cxx.float(),
    )


def observability_spectrum(koopman, reward_map):
    """Singular spectrum of [B; BK; ...; BK^(d-1)]."""
    blocks = []
    block = reward_map.float()
    for _ in range(koopman.shape[0]):
        blocks.append(block)
        block = block @ koopman.float()
    return torch.linalg.svdvals(torch.cat(blocks, dim=0))


class ModelEncoder(nn.Module):
    def __init__(self, obs_dim, hidden, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, latent_dim), std=0.5),
            nn.RMSNorm(latent_dim, elementwise_affine=False),
        )

    def forward(self, raw_obs):
        return self.net(symlog(raw_obs))


class Agent(nn.Module):
    reward_components = 3

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        if args.model_dim < 4:
            raise ValueError("model_dim must include a constant plus at least three latents")
        self.raw_obs_dim = obs_dim
        self.gamma = args.gamma
        self.act_dim = act_dim
        self.model_dim = args.model_dim
        self.latent_dim = args.model_dim - 1
        self.action_basis_dim = 2 * act_dim + act_dim * (act_dim - 1) // 2

        # The policy keeps the reference normalized-observation ThinkTrunk. Model
        # modules are RNG-isolated and never receive actor gradients.
        self.actor_trunk = ThinkTrunk(obs_dim, args.hidden, args.k_blocks, args.n_experts)
        with torch.random.fork_rng(devices=[]):
            self.model_encoder = ModelEncoder(obs_dim, args.model_hidden, self.latent_dim)
            self.reward_action_head = layer_init(
                nn.Linear(self.latent_dim, self.action_basis_dim * self.reward_components),
                std=0.01,
            )
            self.dynamics_action_head = layer_init(
                nn.Linear(self.latent_dim, self.action_basis_dim * self.model_dim),
                std=0.01,
            )
            with torch.no_grad():
                self.reward_action_head.weight.zero_()
                self.reward_action_head.bias.zero_()
                self.dynamics_action_head.weight.zero_()
                self.dynamics_action_head.bias.zero_()
        baseline_rng_dummy = layer_init(nn.Linear(args.hidden, 6 * 511, bias=False), std=0.1)
        del baseline_rng_dummy

        self.actor_alpha_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(args.hidden, act_dim), std=0.01)
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )
        self.register_buffer("koopman", torch.zeros(self.model_dim, self.model_dim))
        self.register_buffer("reward_map", torch.zeros(self.reward_components, self.model_dim))
        self.register_buffer("value_map", torch.zeros(self.reward_components, self.model_dim))
        self.register_buffer("operator_initialized", torch.tensor(False, dtype=torch.bool))

    def encode(self, raw_obs):
        return self.model_encoder(raw_obs)

    def model_state(self, raw_obs):
        latent = self.encode(raw_obs)
        constant = torch.ones((*latent.shape[:-1], 1), device=latent.device, dtype=latent.dtype)
        return torch.cat([constant, latent], dim=-1)

    def value_reasons(self, model_state):
        return model_state @ self.value_map.T

    def scalar_value(self, model_state):
        return self.value_reasons(model_state).sum(dim=-1)

    @torch.no_grad()
    def snapshot_operator(self, koopman, reward_map, value_map):
        discounted_radius = self.gamma * torch.linalg.eigvals(koopman.float()).abs().max()
        matrix = torch.eye(self.model_dim, device=koopman.device) - self.gamma * koopman
        if discounted_radius >= 1.0 or torch.linalg.cond(matrix) > 1e6:
            raise RuntimeError(
                "inadmissible Koopman snapshot: discounted dynamics do not define "
                "a stable resolvent"
            )
        self.koopman.copy_(koopman)
        self.reward_map.copy_(reward_map)
        self.value_map.copy_(value_map)
        self.operator_initialized.fill_(True)

    def _beta(self, actor_feat):
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        return Beta(alpha, beta)

    @staticmethod
    def _centered_beta_monomials(z, alpha, beta):
        """Policy-invariant degree-1/2 monomials with exact Beta centering.

        Only a policy-dependent constant is removed. There is deliberately no
        variance normalization or Gram-Schmidt mixing: a coefficient always
        multiplies the same raw polynomial as alpha/beta change.
        """
        total = alpha + beta
        mean = alpha / total
        second_moment = alpha * (alpha + 1.0) / (total * (total + 1.0))
        linear = z - mean
        quadratic = z.square() - second_moment
        cross_terms = [
            z[:, i] * z[:, j] - mean[:, i] * mean[:, j]
            for i in range(z.shape[-1])
            for j in range(i + 1, z.shape[-1])
        ]
        if cross_terms:
            return torch.cat([linear, quadratic, torch.stack(cross_terms, dim=-1)], dim=-1)
        return torch.cat([linear, quadratic], dim=-1)

    def action_effects(self, latent, basis, value_map=None):
        latent = latent.detach()
        reward_coeff = self.reward_action_head(latent).view(
            -1, self.action_basis_dim, self.reward_components
        )
        dynamics_coeff = self.dynamics_action_head(latent).view(
            -1, self.action_basis_dim, self.model_dim
        )
        reward_effect = torch.einsum("bpr,bp->br", reward_coeff, basis)
        dynamics_effect = torch.einsum("bpd,bp->bd", dynamics_coeff, basis)
        if value_map is None:
            value_map = self.value_map
        reason_advantage = reward_effect + self.gamma * (dynamics_effect @ value_map.T)
        return reward_effect, dynamics_effect, reason_advantage

    def get_action_and_value(self, obs, raw_obs, native_action=None, return_q=False):
        actor_feat = self.actor_trunk(obs)
        dist = self._beta(actor_feat)
        if native_action is None:
            native_action = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * native_action
        log_prob = dist.log_prob(native_action).sum(1)
        entropy = dist.entropy().sum(1)
        model_state = self.model_state(raw_obs)
        value_reasons = self.value_reasons(model_state)
        action_advantage = action_basis = None
        if return_q:
            action_basis = self._centered_beta_monomials(
                native_action, dist.concentration1, dist.concentration0
            )
            _, _, action_advantage = self.action_effects(
                model_state[..., 1:], action_basis
            )
        return (
            action,
            native_action,
            log_prob,
            entropy,
            value_reasons,
            action_advantage,
            action_basis,
        )

    def actor_parameters(self):
        return (
            list(self.actor_trunk.parameters())
            + list(self.actor_alpha_head.parameters())
            + list(self.actor_beta_head.parameters())
        )

    def model_parameters(self):
        return list(self.model_encoder.parameters())

    def action_critic_parameters(self):
        return (
            list(self.reward_action_head.parameters())
            + list(self.dynamics_action_head.parameters())
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert args.batch_size % args.num_minibatches == 0
    assert args.num_envs >= 2, "one environment is reserved for held-out calibration"
    assert args.model_dim >= 4
    assert args.model_epochs > 0 and args.action_epochs > 0
    assert args.operator_ridge > 0.0
    assert 0.0 <= args.qadv_max_blend <= 1.0
    assert args.qadv_slope_max > 0.0
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
        agent.actor_trunk = CompiledTrunk(agent.actor_trunk, args.compile_mode)
        print(f"torch.compile actor trunk mode={args.compile_mode!r}; CUDA graphs disabled")

    actor_optimizer = optim.Adam(agent.actor_parameters(), lr=args.learning_rate, eps=1e-5)
    model_optimizer = optim.Adam(agent.model_parameters(), lr=args.model_lr, eps=1e-5)
    action_optimizer = optim.Adam(agent.action_critic_parameters(), lr=args.action_lr, eps=1e-5)
    actor_params = agent.actor_parameters()
    model_params = agent.model_parameters()
    action_params = agent.action_critic_parameters()

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    raw_obses = torch.zeros(
        (args.num_steps, args.num_envs, agent.raw_obs_dim), device=device
    )
    next_raw_obses = torch.zeros_like(raw_obses)
    native_actions = torch.zeros(
        (args.num_steps, args.num_envs) + action_shape, device=device
    )
    action_bases = torch.zeros(
        (args.num_steps, args.num_envs, agent.action_basis_dim), device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros_like(logprobs)
    reward_components = torch.zeros(
        (args.num_steps, args.num_envs, agent.reward_components), device=device
    )
    values = torch.zeros_like(logprobs)
    reason_values = torch.zeros_like(reward_components)
    predicted_reason_advantages = torch.zeros_like(reward_components)
    transition_terminations = torch.zeros_like(logprobs)
    transition_boundaries = torch.zeros_like(logprobs)
    transition_valids = torch.zeros_like(logprobs)

    global_step = 0
    start_time = time.time()
    next_obs_np, reset_infos = envs.reset(seed=args.seed)
    reset_raw, _, _ = raw_observations_from_infos(reset_infos, args.num_envs)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
    next_raw_obs = torch.as_tensor(reset_raw, device=device, dtype=torch.float32)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            raw_obses[step] = next_raw_obs
            with torch.no_grad():
                action, native_action, logprob, _, value_reason, action_reason, action_basis = (
                    agent.get_action_and_value(next_obs, next_raw_obs, return_q=True)
                )
                reason_values[step] = value_reason
                predicted_reason_advantages[step] = action_reason
                values[step] = value_reason.sum(dim=-1)
            native_actions[step] = native_action
            action_bases[step] = action_basis
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            boundaries = np.logical_or(terminations, truncations)
            emitted_raw, transition_raw, raw_transition_valid = (
                raw_observations_from_infos(infos, args.num_envs, boundaries)
            )
            transition_valid = np.logical_and(~boundaries, raw_transition_valid)
            # A time-limit row becomes bootstrap-valid only with its actual final state.
            transition_valid = np.where(truncations, raw_transition_valid, transition_valid)
            # True termination has a valid zero-bootstrap target even without a final state.
            transition_valid = np.where(terminations, raw_transition_valid, transition_valid)

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            reward_components[step] = torch.as_tensor(
                reward_components_from_infos(infos, args.num_envs),
                device=device,
                dtype=torch.float32,
            )
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                boundaries, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(
                transition_valid, device=device, dtype=torch.float32
            )
            next_raw_obses[step] = torch.as_tensor(
                transition_raw, device=device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_raw_obs = torch.as_tensor(emitted_raw, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            old_states = agent.model_state(
                raw_obses.reshape(-1, agent.raw_obs_dim)
            ).reshape(args.num_steps, args.num_envs, agent.model_dim)
            old_next_states = agent.model_state(
                next_raw_obses.reshape(-1, agent.raw_obs_dim)
            ).reshape(args.num_steps, args.num_envs, agent.model_dim)
            next_reason_values = agent.value_reasons(old_next_states)
            next_values = next_reason_values.sum(dim=-1)
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
            _, reason_returns = compute_scalar_gae(
                reward_components,
                reason_values,
                next_reason_values,
                transition_terminations,
                transition_boundaries,
                transition_valids,
                args.gamma,
                args.gae_lambda,
            )
            reason_sum_error = (reason_returns.sum(dim=-1) - returns).abs()
            reason_sum_tolerance = cancellation_aware_tolerance(reason_returns, atol=1e-4)
            if (reason_sum_error > reason_sum_tolerance).any():
                raise RuntimeError(
                    "reason returns do not sum to scalar returns; "
                    f"max error={reason_sum_error.max().item():.6g}"
                )

            bootstrap = ((1.0 - transition_terminations) * transition_valids).unsqueeze(-1)
            successor_valid = torch.logical_or(
                transition_terminations.bool(), transition_valids.bool()
            )
            old_targets = bootstrap * old_next_states
            old_reason_td = (
                reward_components
                + args.gamma * bootstrap * next_reason_values
                - reason_values
            )

            flat_valid = successor_valid.reshape(-1)
            flat_old_states = old_states.reshape(-1, agent.model_dim)
            flat_old_targets = old_targets.reshape(-1, agent.model_dim)
            flat_rewards_reason = reward_components.reshape(-1, agent.reward_components)
            fit_k, fit_b, fit_g, fit_condition, _ = fit_koopman_resolvent(
                flat_old_states[flat_valid],
                flat_old_targets[flat_valid],
                flat_rewards_reason[flat_valid],
                args.gamma,
                args.operator_ridge,
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
        b_raw_obs = raw_obses.reshape(-1, agent.raw_obs_dim)
        b_next_raw_obs = next_raw_obses.reshape(-1, agent.raw_obs_dim)
        b_native_actions = native_actions.reshape((-1,) + action_shape)
        b_action_bases = action_bases.reshape(-1, agent.action_basis_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_reason_values = reason_values.reshape(-1, agent.reward_components)
        b_predicted_reason_advantages = predicted_reason_advantages.reshape(
            -1, agent.reward_components
        )
        b_old_states = old_states.reshape(-1, agent.model_dim)
        b_old_targets = old_targets.reshape(-1, agent.model_dim)
        b_reward_components = reward_components.reshape(-1, agent.reward_components)
        b_old_reason_td = old_reason_td.reshape(-1, agent.reward_components)
        b_successor_valid = successor_valid.reshape(-1)
        b_truncated_mc_returns = truncated_mc_returns.reshape(-1)
        b_truncated_mc_valid = truncated_mc_valid.reshape(-1)
        row_env_index = torch.arange(args.batch_size, device=device) % args.num_envs
        action_train_mask = b_successor_valid & (row_env_index != 0)
        action_holdout_mask = b_successor_valid & (row_env_index == 0)
        if not action_train_mask.any():
            action_train_mask = b_successor_valid

        with torch.no_grad():
            rollout_qadv = b_predicted_reason_advantages.sum(dim=-1)
            if action_holdout_mask.any():
                calibration_prediction = rollout_qadv[action_holdout_mask]
                calibration_gae = b_advantages[action_holdout_mask]
                centered_calibration_prediction = (
                    calibration_prediction - calibration_prediction.mean()
                )
                centered_calibration_gae = calibration_gae - calibration_gae.mean()
                calibration_slope = (
                    (centered_calibration_gae * centered_calibration_prediction).mean()
                    / centered_calibration_prediction.square().mean().clamp_min(1e-8)
                ).clamp(0.0, args.qadv_slope_max)
                calibration_corr = correlation(calibration_prediction, calibration_gae)
                calibration_blend = (
                    args.qadv_max_blend
                    * calibration_corr.clamp(0.0, 1.0).square()
                )
            else:
                calibration_slope = torch.zeros((), device=device)
                calibration_corr = torch.zeros((), device=device)
                calibration_blend = torch.zeros((), device=device)
            b_policy_advantages = b_advantages.clone()
            b_policy_advantages[b_successor_valid] = (
                (1.0 - calibration_blend) * b_advantages[b_successor_valid]
                + calibration_blend
                * calibration_slope
                * rollout_qadv[b_successor_valid]
            )
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_adv_normed = torch.zeros_like(b_advantages)
            valid_policy_advantages = b_policy_advantages[b_successor_valid]
            b_adv_normed[b_successor_valid] = (
                valid_policy_advantages - valid_policy_advantages.mean()
            ) / (valid_policy_advantages.std() + 1e-8)

        # PPO changes only the policy. The model snapshot that produced this
        # rollout remains frozen throughout policy optimization.
        indices = np.arange(args.batch_size)
        clipfracs = []
        actor_loss_sum = actor_grad_sum = 0.0
        actor_updates = 0
        epochs_completed = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = indices[start : start + args.minibatch_size]
                mb_valid = b_successor_valid[mb].float()
                _, _, newlogprob, entropy, _, _, _ = agent.get_action_and_value(
                    b_obs[mb], b_raw_obs[mb], b_native_actions[mb], return_q=False
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_adv_normed[mb]
                    else:
                        mb_advantages = torch.zeros_like(b_advantages[mb])
                        valid_mb = b_policy_advantages[mb][mb_valid.bool()]
                        if valid_mb.numel() > 1:
                            mb_advantages[mb_valid.bool()] = (
                                valid_mb - valid_mb.mean()
                            ) / (valid_mb.std(unbiased=False) + 1e-8)
                else:
                    mb_advantages = b_policy_advantages[mb]
                clip_high = (
                    args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                )
                pg_loss = torch.maximum(
                    -mb_advantages * ratio,
                    -mb_advantages
                    * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + clip_high),
                )
                pg_loss = masked_mean(pg_loss, mb_valid)
                entropy_loss = masked_mean(entropy, mb_valid)
                actor_optimizer.zero_grad(set_to_none=True)
                (pg_loss - args.ent_coef * entropy_loss).backward()
                actor_grad = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                actor_optimizer.step()
                actor_loss_sum += pg_loss.item()
                actor_grad_sum += float(actor_grad)
                actor_updates += 1
            epochs_completed = epoch + 1
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Alternating representation step. Targets and linear maps are frozen in
        # the old rollout frame; only current-state encoder outputs receive gradients.
        model_indices = np.flatnonzero(b_successor_valid.cpu().numpy())
        reward_scale = (
            b_reward_components[b_successor_valid]
            .std(dim=0, unbiased=False)
            .detach()
            .clamp_min(1.0)
        )
        td_scale = (
            b_old_reason_td[b_successor_valid]
            .std(dim=0, unbiased=False)
            .detach()
            .clamp_min(1.0)
        )
        model_acc = {"reward": 0.0, "closure": 0.0, "td": 0.0, "white": 0.0, "grad": 0.0, "n": 0}
        for _ in range(args.model_epochs):
            np.random.shuffle(model_indices)
            for start in range(0, len(model_indices), args.minibatch_size):
                mb = model_indices[start : start + args.minibatch_size]
                current_state = agent.model_state(b_raw_obs[mb])
                predicted_next = current_state @ fit_k.T
                predicted_reward = current_state @ fit_b.T
                closure_loss = (predicted_next - b_old_targets[mb]).square().mean()
                reward_loss = (
                    (predicted_reward - b_reward_components[mb]) / reward_scale
                ).square().mean()
                # Exact resolvent identity: e_TD=e_r+gamma G e_x. This
                # concentrates representation learning on closure errors that
                # can actually affect future reward.
                reward_innovation = b_reward_components[mb] - predicted_reward
                state_innovation = b_old_targets[mb] - predicted_next
                reason_td_innovation = (
                    reward_innovation + args.gamma * (state_innovation @ fit_g.T)
                )
                td_loss = (reason_td_innovation / td_scale).square().mean()
                white_loss = whitening_loss(current_state[:, 1:])
                model_loss = (
                    args.model_closure_coef * closure_loss
                    + args.model_reward_coef * reward_loss
                    + args.model_td_coef * td_loss
                    + args.model_whiten_coef * white_loss
                )
                model_optimizer.zero_grad(set_to_none=True)
                model_loss.backward()
                model_grad = nn.utils.clip_grad_norm_(model_params, args.critic_grad_clip)
                model_optimizer.step()
                model_acc["reward"] += reward_loss.item()
                model_acc["closure"] += closure_loss.item()
                model_acc["td"] += td_loss.item()
                model_acc["white"] += white_loss.item()
                model_acc["grad"] += float(model_grad)
                model_acc["n"] += 1

        # Fit the exact next-rollout snapshot in the updated encoder frame.
        with torch.no_grad():
            updated_states = agent.model_state(b_raw_obs)
            updated_next_states = agent.model_state(b_next_raw_obs)
            updated_targets = (
                ((1.0 - transition_terminations).reshape(-1, 1)
                 * transition_valids.reshape(-1, 1))
                * updated_next_states
            )
            new_k, new_b, new_g, resolvent_condition, cxx = fit_koopman_resolvent(
                updated_states[b_successor_valid],
                updated_targets[b_successor_valid],
                b_reward_components[b_successor_valid],
                args.gamma,
                args.operator_ridge,
            )
            agent.snapshot_operator(new_k, new_b, new_g)
            # These residuals and the operator frame are now an immutable,
            # mutually consistent snapshot for the next rollout's action field.
            action_reward_targets = (
                b_reward_components - updated_states @ new_b.T
            )
            action_dynamics_targets = updated_targets - updated_states @ new_k.T

        # Fit separate action innovations on envs 1..N-1. Env 0 remains genuinely
        # held out and calibrates this exact frozen predictor on the next rollout.
        action_indices = np.flatnonzero(action_train_mask.cpu().numpy())
        action_reward_scale = (
            action_reward_targets[action_train_mask]
            .std(dim=0, unbiased=False)
            .clamp_min(1.0)
        )
        action_dynamics_scale = (
            action_dynamics_targets[action_train_mask]
            .std(dim=0, unbiased=False)
            .clamp_min(0.1)
        )
        action_acc = {"reward": 0.0, "dynamics": 0.0, "grad": 0.0, "n": 0}
        for _ in range(args.action_epochs):
            np.random.shuffle(action_indices)
            for start in range(0, len(action_indices), args.minibatch_size):
                mb = action_indices[start : start + args.minibatch_size]
                latent = agent.encode(b_raw_obs[mb]).detach()
                reward_prediction, dynamics_prediction, _ = agent.action_effects(
                    latent, b_action_bases[mb], new_g
                )
                reward_action_loss = (
                    (reward_prediction - action_reward_targets[mb])
                    / action_reward_scale
                ).square().mean()
                dynamics_action_loss = (
                    (dynamics_prediction - action_dynamics_targets[mb])
                    / action_dynamics_scale
                ).square().mean()
                loss = reward_action_loss + dynamics_action_loss
                action_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                action_grad = nn.utils.clip_grad_norm_(action_params, args.critic_grad_clip)
                action_optimizer.step()
                action_acc["reward"] += reward_action_loss.item()
                action_acc["dynamics"] += dynamics_action_loss.item()
                action_acc["grad"] += float(action_grad)
                action_acc["n"] += 1

        with torch.no_grad():
            valid = b_successor_valid
            holdout = action_holdout_mask
            diagnostic_train = action_train_mask
            eval_mask = holdout if holdout.any() else valid
            diagnostic_k, diagnostic_b, diagnostic_g, heldout_condition, _ = (
                fit_koopman_resolvent(
                    updated_states[diagnostic_train],
                    updated_targets[diagnostic_train],
                    b_reward_components[diagnostic_train],
                    args.gamma,
                    args.operator_ridge,
                )
            )
            eval_x = updated_states[eval_mask]
            eval_y = updated_targets[eval_mask]
            eval_r = b_reward_components[eval_mask]
            reward_error = eval_x @ diagnostic_b.T - eval_r
            closure_error = eval_x @ diagnostic_k.T - eval_y
            future_reason_error = closure_error @ diagnostic_g.T
            reason_td_error = (
                eval_r
                + args.gamma * (eval_y @ diagnostic_g.T)
                - eval_x @ diagnostic_g.T
            )
            spectral_radius = torch.linalg.eigvals(new_k.float()).abs().max()
            latent = updated_states[valid, 1:].float()
            latent_centered = latent - latent.mean(dim=0)
            latent_covariance = latent_centered.T @ latent_centered / latent.shape[0]
            covariance_eigenvalues = torch.linalg.eigvalsh(latent_covariance)
            covariance_rank = (covariance_eigenvalues > 1e-3).sum()
            obs_spectrum = observability_spectrum(new_k, new_b)
            obs_rank = (obs_spectrum > obs_spectrum.max().clamp_min(1e-8) * 1e-4).sum()

            valid_qadv = rollout_qadv[valid]
            valid_gae = b_advantages[valid]
            action_corr = correlation(valid_qadv, valid_gae)
            action_ev = 1.0 - (valid_gae - valid_qadv).var() / valid_gae.var().clamp_min(1e-8)
            (
                postupdate_reward_effect,
                postupdate_dynamics_effect,
                postupdate_reason_advantage,
            ) = agent.action_effects(
                agent.encode(b_raw_obs[eval_mask]), b_action_bases[eval_mask], new_g
            )
            postupdate_holdout_prediction = postupdate_reason_advantage.sum(dim=-1)
            holdout_gae = b_advantages[eval_mask]
            postupdate_action_corr = correlation(postupdate_holdout_prediction, holdout_gae)
            postupdate_action_ev = 1.0 - (
                (holdout_gae - postupdate_holdout_prediction).var()
                / holdout_gae.var().clamp_min(1e-8)
            )
            if b_truncated_mc_valid.any():
                mc_pred = b_values[b_truncated_mc_valid]
                mc_true = b_truncated_mc_returns[b_truncated_mc_valid]
                mc_ev = 1.0 - (mc_true - mc_pred).var() / mc_true.var().clamp_min(1e-8)
            else:
                mc_ev = torch.tensor(float("nan"), device=device)
            explained_variance = 1.0 - (
                (b_returns[valid] - b_values[valid]).var()
                / b_returns[valid].var().clamp_min(1e-8)
            )

        actor_n = max(actor_updates, 1)
        model_n = max(model_acc["n"], 1)
        action_n = max(action_acc["n"], 1)
        writer.add_scalar(
            "charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/policy_loss", actor_loss_sum / actor_n, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/actor_grad_norm", actor_grad_sum / actor_n, global_step)
        writer.add_scalar("losses/model_reward", model_acc["reward"] / model_n, global_step)
        writer.add_scalar("losses/model_closure", model_acc["closure"] / model_n, global_step)
        writer.add_scalar("losses/model_reason_td", model_acc["td"] / model_n, global_step)
        writer.add_scalar("losses/model_whitening", model_acc["white"] / model_n, global_step)
        writer.add_scalar("losses/model_grad_norm", model_acc["grad"] / model_n, global_step)
        writer.add_scalar(
            "losses/action_reward_innovation",
            action_acc["reward"] / action_n,
            global_step,
        )
        writer.add_scalar(
            "losses/action_dynamics_innovation",
            action_acc["dynamics"] / action_n,
            global_step,
        )
        writer.add_scalar("losses/action_grad_norm", action_acc["grad"] / action_n, global_step)
        writer.add_scalar("losses/explained_variance", explained_variance.item(), global_step)
        writer.add_scalar("diagnostics/truncated_mc_explained_variance", mc_ev.item(), global_step)
        writer.add_scalar(
            "diagnostics/heldout_reward_mse",
            reward_error.square().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/heldout_closure_mse",
            closure_error.square().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/heldout_future_reason_mse",
            future_reason_error.square().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/heldout_reason_td_mse",
            reason_td_error.square().mean().item(),
            global_step,
        )
        writer.add_scalar("diagnostics/action_gae_corr", action_corr.item(), global_step)
        writer.add_scalar("diagnostics/action_gae_ev", action_ev.item(), global_step)
        writer.add_scalar(
            "diagnostics/action_reward_effect_std",
            postupdate_reward_effect.sum(dim=-1).std().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/action_future_effect_std",
            (args.gamma * (postupdate_dynamics_effect @ new_g.T)).sum(dim=-1).std().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/action_postupdate_heldout_corr",
            postupdate_action_corr.item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/action_postupdate_heldout_ev",
            postupdate_action_ev.item(),
            global_step,
        )
        writer.add_scalar("diagnostics/qadv_slope", calibration_slope.item(), global_step)
        writer.add_scalar("diagnostics/qadv_blend", calibration_blend.item(), global_step)
        writer.add_scalar("diagnostics/qadv_heldout_corr", calibration_corr.item(), global_step)
        writer.add_scalar("koopman/spectral_radius", spectral_radius.item(), global_step)
        writer.add_scalar("koopman/resolvent_condition", resolvent_condition.item(), global_step)
        writer.add_scalar("koopman/heldout_fit_condition", heldout_condition.item(), global_step)
        writer.add_scalar("koopman/fit_condition_before_update", fit_condition.item(), global_step)
        writer.add_scalar(
            "koopman/covariance_eig_min", covariance_eigenvalues.min().item(), global_step
        )
        writer.add_scalar(
            "koopman/covariance_eig_max", covariance_eigenvalues.max().item(), global_step
        )
        writer.add_scalar("koopman/covariance_rank", covariance_rank.item(), global_step)
        writer.add_scalar("koopman/observability_rank", obs_rank.item(), global_step)
        writer.add_scalar("koopman/observability_sv_min", obs_spectrum.min().item(), global_step)
        writer.add_scalar("koopman/observability_sv_max", obs_spectrum.max().item(), global_step)
        writer.add_scalar("debug/epochs_completed", epochs_completed, global_step)
        writer.add_scalar(
            "debug/reason_return_sum_maxerr", reason_sum_error.max().item(), global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    envs.close()
    writer.close()
