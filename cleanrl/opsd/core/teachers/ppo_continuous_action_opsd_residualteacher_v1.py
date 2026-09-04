# OPSD Residual Teacher v1 -- privileged-context self-distillation with clone isolation.
#
# A shared conditional trunk lets rationalization NLL directly rewrite the zero-context
# actor, so its clone term is an ordinary behavior-cloning actor loss rather than canonical
# OPSD. Here the proven v6 zero-context student is unchanged. A separately optimized,
# zero-initialized residual adapter alone learns p(a | s, realized TD residual); its frozen
# optimistic query is then distilled into the base actor, whose sole actor loss is the
# paper's clipped forward KL. Hypothesis: isolating clone gradients preserves a useful
# privileged channel without clone-driven student sharpening. It is falsified if channel
# nats and teacher/student KL appear but teacher-to-student KL does not fall under the base
# update, or if that reduction does not accompany improved on-policy returns.

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

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport


SAMPLE_EPS = 1e-6


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 1e-3
    num_envs: int = 16
    num_steps: int = 2048
    async_envs: bool = True
    compile_act: bool = True
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 128
    actor_epochs: int = 4
    critic_epochs: int = 4

    adv_boost: float = 1.0
    adv_cond_clip: float = 3.0
    cond_scale: str = "ema_rms"
    cond_ema_beta: float = 0.99
    adv_embed_freqs: int = 8
    clone_coef: float = 1.0
    distill_coef: float = 1.0
    distill_kl_clip: float = 2.0

    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
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
        env = gym.wrappers.TransformObservation(  # pyright: ignore[reportCallIssue]
            env, lambda observation: np.asarray(observation).clip(-10, 10)
        )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(
                env, lambda reward: min(max(float(reward), -10.0), 10.0)
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def beta_kl_per_dim(a1, b1, a2, b2):
    """Forward KL(Beta(a1,b1) || Beta(a2,b2)) for each action dimension."""

    def ln_beta_fn(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    return (
        ln_beta_fn(a2, b2)
        - ln_beta_fn(a1, b1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


def grad_norm(parameters):
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        term = parameter.grad.detach().float().square().sum()
        squared_norm = term if squared_norm is None else squared_norm + term
    return 0.0 if squared_norm is None else squared_norm.sqrt().item()


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(hidden):
    return nn.Sequential(
        layer_init(nn.Linear(hidden, hidden)),
        ReLUSquared(),
        layer_init(nn.Linear(hidden, hidden)),
    )


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

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
        moe_input = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(moe_input), dim=-1)
        expert_outputs = torch.stack([expert(moe_input) for expert in self.experts], dim=1)
        moe_delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return x_in + dense_delta + moe_delta


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, hidden, k_blocks, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, hidden))
        self.blocks = nn.ModuleList(
            [ThinkBlock(hidden * (block + 1), hidden, n_experts) for block in range(k_blocks)]
        )
        cat_dim = hidden * (k_blocks + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, hidden))

    def forward(self, x):
        x0 = self.entry(x)
        features = [x0]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(features, dim=-1)))


class AdvEmbed(nn.Module):
    """Fixed Fourier features for one present, scaled TD residual."""

    def __init__(self, n_freq):
        super().__init__()
        self.n_freq = n_freq
        self.freqs: torch.Tensor
        self.register_buffer("freqs", torch.logspace(-1.0, 3.0, n_freq, base=2.0))

    @property
    def dim(self):
        return 2 * self.n_freq

    def forward(self, advantage):
        phase = advantage * self.freqs
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class PrivilegedResidualAdapter(nn.Module):
    """Maps detached base features and a present context to base-logit residuals."""

    def __init__(self, feature_dim, cond_dim, hidden, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(feature_dim + cond_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden, 2 * act_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, detached_base_features, present_context):
        residual = self.out(self.body(torch.cat([detached_base_features, present_context], dim=-1)))
        return residual.chunk(2, dim=-1)


class Agent(nn.Module):
    """The v6 zero-context base student plus an isolated privileged teacher adapter.

    Absence is represented only by student_policy(), which bypasses the adapter and feeds
    the original all-zero context into the original-shape trunk. teacher_policy() requires
    an explicit [batch, 1] context, including when its numerical value is zero; present zero
    is Fourier-embedded and can never alias absence.
    """

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        hidden = args.hidden
        self.adv_embed = AdvEmbed(args.adv_embed_freqs)
        self.cond_dim = self.adv_embed.dim
        # This input shape and its zero block are exactly the v6 base/student architecture.
        self.trunk = ThinkTrunk(
            obs_dim + self.cond_dim, hidden, args.k_blocks, args.n_experts
        )
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        # Preserve v6's base-module construction order, including its seeded actor init.
        self.critic_head = layer_init(
            nn.Linear(hidden, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.actor_alpha_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        # Adapter initialization must not advance the student's rollout RNG. This keeps
        # the base initialization and first on-policy sample on the proven v6 stream.
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + 1_000_003)
            self.residual_adapter = PrivilegedResidualAdapter(
                hidden, self.cond_dim, hidden, act_dim
            )
        self.action_low: torch.Tensor
        self.action_high: torch.Tensor
        self.register_buffer(
            "action_low",
            torch.as_tensor(envs.single_action_space.low, dtype=torch.float32).reshape(-1),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(envs.single_action_space.high, dtype=torch.float32).reshape(-1),
        )
        assert torch.count_nonzero(self.residual_adapter.out.weight).item() == 0
        assert torch.count_nonzero(self.residual_adapter.out.bias).item() == 0

    def _zero_cond(self, obs):
        return obs.new_zeros((obs.shape[0], self.cond_dim))

    def _base_features(self, obs):
        return self.trunk(torch.cat([obs, self._zero_cond(obs)], dim=-1))

    def _base_logits(self, base_features):
        return self.actor_alpha_head(base_features), self.actor_beta_head(base_features)

    def _check_present_context(self, obs, privileged_query):
        if privileged_query is None:
            raise ValueError("teacher_policy requires present privileged context")
        if privileged_query.ndim != 2 or privileged_query.shape != (obs.shape[0], 1):
            raise ValueError(
                "privileged context must have shape "
                f"({obs.shape[0]}, 1), got {tuple(privileged_query.shape)}"
            )

    def student_policy(self, obs):
        base_features = self._base_features(obs)
        alpha_logits, beta_logits = self._base_logits(base_features)
        return 1.0 + F.softplus(alpha_logits), 1.0 + F.softplus(beta_logits)

    def student_policy_and_value(self, obs):
        base_features = self._base_features(obs)
        alpha_logits, beta_logits = self._base_logits(base_features)
        value_logits = self.critic_head(base_features).view(
            -1, self.critic_mtp_horizon, self.num_bins
        )
        return (
            1.0 + F.softplus(alpha_logits),
            1.0 + F.softplus(beta_logits),
            value_logits,
        )

    def _teacher_logits_and_residual(self, obs, privileged_query):
        self._check_present_context(obs, privileged_query)
        # Clone gradients stop at the adapter boundary by construction.
        with torch.no_grad():
            detached_features = self._base_features(obs)
            base_alpha_logits, base_beta_logits = self._base_logits(detached_features)
            present_context = self.adv_embed(privileged_query)
        alpha_residual, beta_residual = self.residual_adapter(
            detached_features, present_context
        )
        return (
            base_alpha_logits + alpha_residual,
            base_beta_logits + beta_residual,
            alpha_residual,
            beta_residual,
        )

    def teacher_policy(self, obs, privileged_query):
        alpha_logits, beta_logits, _, _ = self._teacher_logits_and_residual(
            obs, privileged_query
        )
        return 1.0 + F.softplus(alpha_logits), 1.0 + F.softplus(beta_logits)

    def adapter_residual(self, obs, privileged_query):
        _, _, alpha_residual, beta_residual = self._teacher_logits_and_residual(
            obs, privileged_query
        )
        return alpha_residual, beta_residual

    def act(self, obs):
        alpha, beta, value_logits = self.student_policy_and_value(obs)
        distribution = Beta(alpha, beta, validate_args=False)
        latent = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * latent
        return action, latent, value_logits

    def get_value(self, obs):
        base_features = self._base_features(obs)
        return self.critic_head(base_features).view(
            -1, self.critic_mtp_horizon, self.num_bins
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.env_id == "HalfCheetah-v4", "this versioned arm is HalfCheetah-v4 only"
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.adv_boost > 0.0, "a non-positive margin makes the teacher no better"

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
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    vector_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_cls(
        [
            make_env(
                args.env_id,
                env_index,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for env_index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, args).to(device)
    act_fn = (
        torch.compile(agent.act, mode="reduce-overhead") if args.compile_act else agent.act
    )

    all_params = list(agent.parameters())
    adapter_params = list(agent.residual_adapter.parameters())
    adapter_param_ids = {id(parameter) for parameter in adapter_params}
    base_params = [parameter for parameter in all_params if id(parameter) not in adapter_param_ids]
    base_param_ids = {id(parameter) for parameter in base_params}
    all_param_ids = {id(parameter) for parameter in all_params}
    assert adapter_param_ids.isdisjoint(base_param_ids)
    assert adapter_param_ids | base_param_ids == all_param_ids
    assert len(adapter_param_ids) + len(base_param_ids) == len(all_param_ids)

    base_actor_params = list(agent.trunk.parameters()) + list(
        agent.actor_alpha_head.parameters()
    ) + list(agent.actor_beta_head.parameters())
    assert {id(parameter) for parameter in base_actor_params} <= base_param_ids

    base_optimizer = optim.Adam(base_params, lr=args.learning_rate, eps=1e-5)
    adapter_optimizer = optim.Adam(adapter_params, lr=args.learning_rate, eps=1e-5)
    assert {
        id(parameter)
        for group in base_optimizer.param_groups
        for parameter in group["params"]
    } == base_param_ids
    assert {
        id(parameter)
        for group in adapter_optimizer.param_groups
        for parameter in group["params"]
    } == adapter_param_ids

    # This permutation stream is isolated from action sampling and base minibatching.
    adapter_generator = torch.Generator(device=device)
    adapter_generator.manual_seed(args.seed + 1_000_003)

    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs), device=device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    cond_ms = torch.zeros((), device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            learning_rate = (
                1.0 - (iteration - 1.0) / args.num_iterations
            ) * args.learning_rate
            base_optimizer.param_groups[0]["lr"] = learning_rate
            adapter_optimizer.param_groups[0]["lr"] = learning_rate

        # Fresh actions always come from the unprivileged base student.
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, latent, value_logits = act_fn(next_obs)
                values[step] = hl_support.to_scalar(value_logits[:, 0])
            latent_zs[step] = latent

            env_action = action.reshape((args.num_envs,) + action_shape)
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [item is not None for item in final_obs]
                for env_index, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_index] is not None:
                        transition_next_obs[env_index] = final_obs[env_index]
                        transition_valid[env_index] = 1.0
                    elif transition_boundary[env_index]:
                        transition_valid[env_index] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
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
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # GAE and HL-Gauss MTP targets stay identical to the v6 state-value critic.
        with torch.no_grad():
            next_value_logits = agent.get_value(next_obses.reshape((-1,) + obs_shape))[:, 0]
            next_values = hl_support.to_scalar(next_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            advantages = torch.zeros_like(rewards)
            td_residuals = torch.zeros_like(rewards)
            last_gae = torch.zeros(args.num_envs, device=device)
            for step in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (
                    1.0 - transition_terminations[step]
                ) * transition_valids[step]
                lambda_nonterminal = 1.0 - transition_boundaries[step]
                delta = (
                    rewards[step]
                    + args.gamma * next_values[step] * bootstrap_nonterminal
                    - values[step]
                )
                td_residuals[step] = delta
                last_gae = (
                    delta
                    + args.gamma * args.gae_lambda * lambda_nonterminal * last_gae
                )
                advantages[step] = last_gae
            returns = advantages + values

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=device
            )
            for horizon in range(mtp):
                valid_len = args.num_steps - horizon
                valid_horizon = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=device
                )
                for boundary_offset in range(horizon):
                    valid_horizon &= (
                        transition_boundaries[
                            boundary_offset : boundary_offset + valid_len
                        ]
                        == 0
                    )
                return_mtp[:valid_len, :, horizon] = returns[horizon:]
                return_mtp_mask[:valid_len, :, horizon] = valid_horizon
            target_probs = hl_support.project(return_mtp)

            # The sole privileged datum is the realized one-step TD residual.
            scaled_residual = td_residuals.reshape(-1)
            if args.cond_scale == "batch":
                scaled_residual = (scaled_residual - scaled_residual.mean()) / (
                    scaled_residual.std() + 1e-8
                )
            elif args.cond_scale == "ema_rms":
                residual_ms = scaled_residual.square().mean()
                cond_ms.mul_(args.cond_ema_beta).add_(
                    (1.0 - args.cond_ema_beta) * residual_ms
                )
                bias_correction = 1.0 - args.cond_ema_beta**iteration
                scaled_residual = scaled_residual / (
                    cond_ms / bias_correction
                ).sqrt().clamp_min(1e-8)
            elif args.cond_scale != "raw":
                raise ValueError(f"unknown cond_scale {args.cond_scale!r}")
            cond_scale_used = scaled_residual.square().mean().sqrt().item()
            b_privileged = scaled_residual.clamp(
                -args.adv_cond_clip, args.adv_cond_clip
            ).unsqueeze(-1)
            cond_clip_frac = (
                scaled_residual.abs() >= args.adv_cond_clip
            ).float().mean().item()
            query_all = (b_privileged + args.adv_boost).clamp(
                -args.adv_cond_clip, args.adv_cond_clip
            )
            query_clip_frac = (
                query_all.abs() >= args.adv_cond_clip - 1e-6
            ).float().mean().item()

        b_obs = obs.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(
            -1, args.critic_mtp_horizon, args.num_bins
        )
        b_target_mask = return_mtp_mask.reshape(
            -1, args.critic_mtp_horizon
        ).to(torch.float32)

        # 1. Rationalize one fresh-data pass. Only adapter parameters may receive gradients.
        adapter_clone_losses = []
        student_nlls = []
        adapter_grad_norms = []
        base_optimizer.zero_grad(set_to_none=True)
        adapter_permutation = torch.randperm(
            args.batch_size, device=device, generator=adapter_generator
        )
        for start in range(0, args.batch_size, args.minibatch_size):
            minibatch = adapter_permutation[start : start + args.minibatch_size]
            teacher_alpha, teacher_beta = agent.teacher_policy(
                b_obs[minibatch], b_privileged[minibatch]
            )
            clone_nll = -Beta(
                teacher_alpha, teacher_beta, validate_args=False
            ).log_prob(b_z[minibatch]).sum(-1).mean()
            with torch.no_grad():
                student_alpha, student_beta = agent.student_policy(b_obs[minibatch])
                student_nll = -Beta(
                    student_alpha, student_beta, validate_args=False
                ).log_prob(b_z[minibatch]).sum(-1).mean()

            adapter_optimizer.zero_grad(set_to_none=True)
            (args.clone_coef * clone_nll).backward()
            assert all(parameter.grad is None for parameter in base_params)
            adapter_grad_norms.append(
                float(
                    nn.utils.clip_grad_norm_(adapter_params, args.max_grad_norm).item()
                )
            )
            adapter_optimizer.step()
            adapter_clone_losses.append(clone_nll.item())
            student_nlls.append(student_nll.item())

        assert all(parameter.grad is None for parameter in base_params)

        # 2. Freeze the rationalized optimistic teacher and the pre-update student once.
        target_alphas = []
        target_betas = []
        old_alphas = []
        old_betas = []
        realized_kls = []
        realized_nlls = []
        query_kls = []
        residual_rms_chunks = []
        teacher_entropies = []
        cond_gaps = []
        with torch.no_grad():
            for start in range(0, args.batch_size, args.minibatch_size):
                batch_slice = slice(start, start + args.minibatch_size)
                old_alpha, old_beta = agent.student_policy(b_obs[batch_slice])
                realized_alpha, realized_beta = agent.teacher_policy(
                    b_obs[batch_slice], b_privileged[batch_slice]
                )
                target_alpha, target_beta = agent.teacher_policy(
                    b_obs[batch_slice], query_all[batch_slice]
                )
                alpha_residual, beta_residual = agent.adapter_residual(
                    b_obs[batch_slice], query_all[batch_slice]
                )

                target_alphas.append(target_alpha.detach())
                target_betas.append(target_beta.detach())
                old_alphas.append(old_alpha.detach())
                old_betas.append(old_beta.detach())
                realized_nlls.append(
                    -Beta(
                        realized_alpha, realized_beta, validate_args=False
                    ).log_prob(b_z[batch_slice]).sum(-1).mean().item()
                )
                realized_kls.append(
                    beta_kl_per_dim(
                        realized_alpha, realized_beta, old_alpha, old_beta
                    ).clamp_min(0.0).sum(-1).mean().item()
                )
                query_kls.append(
                    beta_kl_per_dim(
                        target_alpha, target_beta, old_alpha, old_beta
                    ).clamp_min(0.0).sum(-1).mean().item()
                )
                residual_rms_chunks.append(
                    torch.cat([alpha_residual, beta_residual], dim=-1)
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                )
                teacher_entropies.append(
                    Beta(target_alpha, target_beta, validate_args=False)
                    .entropy()
                    .sum(-1)
                    .mean()
                    .item()
                )
                cond_gaps.append(
                    (
                        target_alpha / (target_alpha + target_beta)
                        - old_alpha / (old_alpha + old_beta)
                    ).abs().mean().item()
                )

        b_target_alpha = torch.cat(target_alphas).detach()
        b_target_beta = torch.cat(target_betas).detach()
        b_old_alpha = torch.cat(old_alphas).detach()
        b_old_beta = torch.cat(old_betas).detach()
        frozen_targets = (
            b_target_alpha,
            b_target_beta,
            b_old_alpha,
            b_old_beta,
        )
        assert all(
            not target.requires_grad and target.grad_fn is None
            for target in frozen_targets
        )
        adapter_optimizer.zero_grad(set_to_none=True)

        # 3. The base actor receives only clipped teacher->student distillation gradients.
        distill_kls = []
        value_losses = []
        actor_grad_norms = []
        for epoch in range(max(args.actor_epochs, args.critic_epochs)):
            do_actor = epoch < args.actor_epochs
            do_critic = epoch < args.critic_epochs
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                minibatch = permutation[start : start + args.minibatch_size]
                base_optimizer.zero_grad(set_to_none=True)

                if do_actor and do_critic:
                    student_alpha, student_beta, value_logits = (
                        agent.student_policy_and_value(b_obs[minibatch])
                    )
                elif do_actor:
                    student_alpha, student_beta = agent.student_policy(b_obs[minibatch])
                    value_logits = None
                else:
                    student_alpha = student_beta = None
                    value_logits = agent.get_value(b_obs[minibatch])

                if do_actor:
                    target_alpha = b_target_alpha[minibatch]
                    target_beta = b_target_beta[minibatch]
                    assert not target_alpha.requires_grad and not target_beta.requires_grad
                    kl_per_dim = beta_kl_per_dim(
                        target_alpha,
                        target_beta,
                        student_alpha,
                        student_beta,
                    ).clamp_min(0.0)
                    distill_loss = kl_per_dim.clamp(
                        max=args.distill_kl_clip
                    ).sum(-1).mean()
                    (args.distill_coef * distill_loss).backward(
                        retain_graph=do_critic
                    )
                    actor_grad_norms.append(grad_norm(base_actor_params))
                    distill_kls.append(kl_per_dim.sum(-1).mean().item())

                if do_critic:
                    assert value_logits is not None
                    log_value_probs = torch.log_softmax(value_logits, dim=-1)
                    value_cross_entropy = -(
                        b_target_probs[minibatch] * log_value_probs
                    ).sum(dim=-1)
                    value_loss = (
                        value_cross_entropy * b_target_mask[minibatch]
                    ).sum(dim=-1).mean()
                    (args.vf_coef * value_loss).backward()
                    value_losses.append(value_loss.item())

                assert all(parameter.grad is None for parameter in adapter_params)
                nn.utils.clip_grad_norm_(base_params, args.max_grad_norm)
                base_optimizer.step()

        # Fixed-target post-update diagnostics distinguish movement from target tracking.
        old_new_kls = []
        teacher_new_kls = []
        student_entropies = []
        with torch.no_grad():
            for start in range(0, args.batch_size, args.minibatch_size):
                batch_slice = slice(start, start + args.minibatch_size)
                new_alpha, new_beta = agent.student_policy(b_obs[batch_slice])
                old_new_kls.append(
                    beta_kl_per_dim(
                        b_old_alpha[batch_slice],
                        b_old_beta[batch_slice],
                        new_alpha,
                        new_beta,
                    ).clamp_min(0.0).sum(-1).mean().item()
                )
                teacher_new_kls.append(
                    beta_kl_per_dim(
                        b_target_alpha[batch_slice],
                        b_target_beta[batch_slice],
                        new_alpha,
                        new_beta,
                    ).clamp_min(0.0).sum(-1).mean().item()
                )
                student_entropies.append(
                    Beta(new_alpha, new_beta, validate_args=False)
                    .entropy()
                    .sum(-1)
                    .mean()
                    .item()
                )

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance = np.var(y_true)
        explained_variance = (
            np.nan
            if variance == 0
            else 1 - np.var(y_true - y_pred) / variance
        )
        sps = int(global_step / (time.time() - start_time))

        adapter_clone_nll = float(np.mean(realized_nlls))
        student_nll = float(np.mean(student_nlls))
        query_teacher_student_kl = float(np.mean(query_kls))
        teacher_new_student_kl = float(np.mean(teacher_new_kls))

        writer.add_scalar(
            "charts/learning_rate", base_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "charts/adapter_learning_rate",
            adapter_optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/adapter_clone_nll", adapter_clone_nll, global_step)
        writer.add_scalar(
            "losses/adapter_train_nll", float(np.mean(adapter_clone_losses)), global_step
        )
        writer.add_scalar("losses/student_nll", student_nll, global_step)
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/value_loss", float(np.mean(value_losses)), global_step)
        writer.add_scalar(
            "losses/adapter_grad_norm", float(np.mean(adapter_grad_norms)), global_step
        )
        writer.add_scalar(
            "losses/actor_grad_norm", float(np.mean(actor_grad_norms)), global_step
        )
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/channel_nats", student_nll - adapter_clone_nll, global_step)
        writer.add_scalar(
            "debug/realized_teacher_student_kl",
            float(np.mean(realized_kls)),
            global_step,
        )
        writer.add_scalar(
            "debug/query_teacher_student_kl", query_teacher_student_kl, global_step
        )
        writer.add_scalar(
            "debug/adapter_residual_rms",
            float(np.mean(residual_rms_chunks)),
            global_step,
        )
        writer.add_scalar(
            "debug/student_entropy", float(np.mean(student_entropies)), global_step
        )
        writer.add_scalar(
            "debug/old_student_new_student_kl",
            float(np.mean(old_new_kls)),
            global_step,
        )
        writer.add_scalar(
            "debug/teacher_new_student_kl", teacher_new_student_kl, global_step
        )
        writer.add_scalar(
            "debug/teacher_student_kl_reduction",
            query_teacher_student_kl - teacher_new_student_kl,
            global_step,
        )
        writer.add_scalar("debug/query_clip_frac", query_clip_frac, global_step)
        writer.add_scalar("debug/cond_scale_rms", cond_scale_used, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clip_frac, global_step)
        writer.add_scalar("debug/cond_gap", float(np.mean(cond_gaps)), global_step)
        writer.add_scalar(
            "debug/teacher_entropy", float(np.mean(teacher_entropies)), global_step
        )
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()
