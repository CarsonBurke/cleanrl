# OPSD Residual-Teacher + Jacobian Critic-Split v2
# =====================================================================================
# v1 established that a held-out one-step pathwise Jacobian adds useful, nearly orthogonal
# credit direction, but its shared value update moved the final student away from the frozen
# teacher. This version gives actor and critic disjoint trunks and optimizers. The critic
# trunk starts as an exact copy of the actor trunk, so initial policy and value predictions
# are unchanged; thereafter value regression cannot overwrite a distilled actor feature.
#
# The canonical OPSD teacher remains the sole actor driver. A privileged residual adapter
# learns p(a_t | s_t, delta_t) from detached actor features, then its frozen optimistic query
# is distilled into the zero-context actor. The one-step TransitionHead rotates that target
# toward d[gamma V(s')]/dz at exact Fisher/std cosine 0.95 and matched clipped-KL dose.
#
# Clone NLL owns only adapter parameters, distillation owns only actor parameters, and
# HL-Gauss regression owns only critic parameters. There is no PPO ratio, surrogate,
# advantage-weighted actor gradient, Q critic, candidate search, counterfactual rollout,
# extra teacher pass, or added teacher dose. Existing fixed-teacher KL-reduction telemetry is
# the primary test: critic isolation should turn the v1 negative reduction positive while
# retaining Jacobian R2, rotation, and matched-dose invariants.
# =====================================================================================
import copy
import math
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
    # --- throughput. Neither flag changes the algorithm: identical math, identical
    # per-env seeding. Measured on this chassis: the acting forward is LAUNCH-bound, not
    # compute-bound (eager act = 1149 us at batch 16, 1277 us at batch 256 -- flat), and it
    # dominated the 5.8 ms vec-step while env stepping was only 0.99 ms of it.
    #   compile mode=reduce-overhead: 1149 -> 347 us at batch 16.
    #   AsyncVectorEnv: raw env stepping 16131 -> 40357 samples/s.
    #   marginal end-to-end: 3025 -> 4300 SPS at 16 envs; 6013 at 128 envs.
    # NOTE the 8819 @8M champion was measured EAGER; arms after this patch are compiled, so
    # numerics are not bit-identical (well inside the +-498 CI95 on that measurement).
    async_envs: bool = True
    compile_act: bool = True
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 128
    actor_epochs: int = 4         # repeated distillation passes over the frozen teacher
    critic_epochs: int = 4        # supervised passes over fixed HL-Gauss value targets

    # --- OPSD advantage conditioning ---
    adv_boost: float = 1.0        # margin, in cond-scale units, each state must beat ITSELF by
    adv_boost_final: float = -1.0 # if >= 0, linearly anneal adv_boost -> this over training.
                                  # Negative (default) = constant margin, i.e. bit-for-bit
                                  # the previous behaviour.
                                  # Motivation, measured at 2M on this chassis:
                                  #   kappa   @500k    @1M     @2M
                                  #     1       664    2962    5365
                                  #     2       924    3693    6077   <- stable optimum
                                  #     3      1181    3691    5624
                                  #     4      1600    4255    3128   <- fastest, then broke
                                  # A big margin buys the fastest early progress in the
                                  # lineage and then destabilizes; a small one is stable but
                                  # slow. The margin is a step size, so decay it.
    adv_cond_clip: float = 3.0    # clamp on the scaled advantage used as conditioning
    cond_scale: str = "ema_rms"   # "ema_rms" | "batch" | "raw"; see the scaling block below
    cond_ema_beta: float = 0.99   # EMA horizon for the RMS scale (~100 iterations)
    adv_embed_freqs: int = 8      # sinusoidal features per phase; privileged block is 2x this
    cond_lambda: float = 0.0      # GAE lambda for the PRIVILEGED channel only; 0 = 1-step
                                  # TD residual, the most action-attributable credit signal
    clone_coef: float = 1.0       # adapter-only realized-context rationalization weight
    distill_coef: float = 1.0     # weight on the per-dim teacher->student divergence
    distill_kl_clip: float = 2.0  # tau: the paper's per-token pointwise divergence clip

    # --- held-out-R2-gated pathwise rotation of the residual OPSD teacher -------------
    jac_cos: float = 0.95       # Fisher-metric cosine retained with the OPSD displacement
    jac_hidden: int = 64        # one-step TransitionHead width
    jac_coef: float = 1.0       # transition regression weight
    jac_dose_iters: int = 16    # clipped-KL dose-match bisection iterations
    jac_r2_min: float = 0.5     # gate fully closed at or below this held-out R2
    jac_r2_open: float = 0.8    # gate fully open at or above this held-out R2

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
    """Forward KL( Beta(a1,b1) || Beta(a2,b2) ) per action dim, closed form."""

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


def _branch_body(H):
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
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in))
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            self.blocks.append(ThinkBlock(H * (k + 1), H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class AdvEmbed(nn.Module):
    """Fixed Fourier features of the scalar advantage.

    A single raw advantage channel among 17 observation dims is trivially IGNORABLE: the
    rationalization loss can be driven down almost entirely by modelling the marginal
    p(a|s) and dropping A, because the extra likelihood A buys is small. Measured on
    HalfCheetah with raw scalar conditioning: cond_gap 0.001 and distill_kl 0.0001, i.e.
    the teacher WAS the student and the method was a no-op. Sinusoidal features of the
    scalar fix this the way diffusion timestep embeddings do -- the advantage now occupies
    many channels and separates nearby values at high frequency, so it is both easy to use
    and expensive to ignore. Frequencies are fixed, not learned, so the channel cannot be
    switched off by driving weights to zero.
    """

    def __init__(self, n_freq):
        super().__init__()
        self.n_freq = n_freq
        self.freqs: torch.Tensor
        self.register_buffer("freqs", torch.logspace(-1.0, 3.0, n_freq, base=2.0))

    @property
    def dim(self):
        return 2 * self.n_freq

    def forward(self, adv):
        x = adv * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class TransitionHead(nn.Module):
    """One-step action-conditioned state-delta model: g(s, a) -> standardized ds.

    DELIBERATELY NOT ON THE SHARED TRUNK. Routing model gradients through the actor's
    representation would confound "the analytic direction helps" with "predicting the
    next state is a good auxiliary task", and the family has already been burned by
    exactly that class of confound. This is a separate 2-layer MLP on raw obs + action;
    at hidden=64 it is a rounding error next to the env loop.

    Targets are standardized per dimension with running statistics, because the qpos and
    qvel blocks of a MuJoCo observation differ by ~26x in action sensitivity (measured
    median |dJ| 0.0052 vs 0.1359) and an unstandardized MSE would fit only the velocities.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, obs_dim), std=0.01),
        )
        self.ds_mean: torch.Tensor
        self.ds_var: torch.Tensor
        self.register_buffer("ds_mean", torch.zeros(obs_dim))
        self.register_buffer("ds_var", torch.ones(obs_dim))

    def forward(self, obs, action):
        """Standardized delta prediction (the space the regression loss lives in)."""
        return self.net(torch.cat([obs, action], dim=-1))

    def next_obs(self, obs, action):
        """Predicted next observation in RAW units -- what V(s') must be evaluated on."""
        ds = self.forward(obs, action) * self.ds_var.sqrt().clamp_min(1e-6) + self.ds_mean
        return obs + ds

    @torch.no_grad()
    def update_stats(self, ds, beta):
        self.ds_mean.mul_(beta).add_((1.0 - beta) * ds.mean(0))
        self.ds_var.mul_(beta).add_((1.0 - beta) * ds.var(0))


class PrivilegedResidualAdapter(nn.Module):
    """Additive pre-softplus policy-logit residual for a present privileged context."""

    def __init__(self, feat_dim, cond_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(feat_dim + cond_dim, feat_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(feat_dim, feat_dim)),
            nn.SiLU(),
        )
        self.out = nn.Linear(feat_dim, 2 * act_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, detached_base_feat, present_adv_embed):
        residual = self.out(self.body(torch.cat([detached_base_feat, present_adv_embed], dim=-1)))
        return residual.chunk(2, dim=-1)


class Agent(nn.Module):
    """Unchanged zero-context student plus an explicitly present-context adapter."""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.adv_embed = AdvEmbed(args.adv_embed_freqs)
        self.cond_dim = self.adv_embed.dim
        # This is the original v6 ThinkTrunk shape. The student always supplies its exact
        # zero block; the adapter is outside the trunk and cannot alter this architecture.
        self.trunk = ThinkTrunk(obs_dim + self.cond_dim, H, args.k_blocks, args.n_experts)
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        # Exact-copy initialization isolates ownership without changing the initial value
        # function or consuming RNG that would perturb the on-policy action sequence.
        self.critic_trunk = copy.deepcopy(self.trunk)
        assert all(
            torch.equal(actor_param, critic_param)
            for actor_param, critic_param in zip(
                self.trunk.parameters(), self.critic_trunk.parameters(), strict=True
            )
        )
        # Adapter initialization is isolated so adding the privileged teacher cannot shift
        # the base student's initialization or on-policy action-sampling RNG.
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + 1_000_003)
            self.residual_adapter = PrivilegedResidualAdapter(H, self.cond_dim, act_dim)
        assert torch.count_nonzero(self.residual_adapter.out.weight).item() == 0
        assert torch.count_nonzero(self.residual_adapter.out.bias).item() == 0
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

    def _zero_cond(self, obs):
        return obs.new_zeros((obs.shape[0], self.cond_dim))

    def _actor_feat(self, obs):
        return self.trunk(torch.cat([obs, self._zero_cond(obs)], dim=-1))

    def _critic_feat(self, obs):
        return self.critic_trunk(torch.cat([obs, self._zero_cond(obs)], dim=-1))

    def student_policy(self, obs):
        feat = self._actor_feat(obs)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        return alpha, beta

    def student_policy_and_value(self, obs):
        alpha, beta = self.student_policy(obs)
        value_logits = self.critic_head(self._critic_feat(obs)).view(
            -1, self.critic_mtp_horizon, self.num_bins
        )
        return alpha, beta, value_logits

    def _present_embed(self, privileged_query, batch_size):
        if (
            privileged_query is None
            or privileged_query.ndim != 2
            or privileged_query.shape != (batch_size, 1)
        ):
            raise ValueError(
                "teacher context must be an explicitly present scalar with shape [batch, 1]"
            )
        return self.adv_embed(privileged_query)

    def _detached_teacher_parts(self, obs, privileged_query):
        cond = self._present_embed(privileged_query, obs.shape[0])
        with torch.no_grad():
            feat = self._actor_feat(obs)
            alpha_logit = self.actor_alpha_head(feat)
            beta_logit = self.actor_beta_head(feat)
        alpha_resid, beta_resid = self.residual_adapter(feat, cond)
        return alpha_logit, beta_logit, alpha_resid, beta_resid

    def teacher_policy(self, obs, privileged_query):
        alpha_logit, beta_logit, alpha_resid, beta_resid = self._detached_teacher_parts(
            obs, privileged_query
        )
        alpha = 1.0 + F.softplus(alpha_logit + alpha_resid)
        beta = 1.0 + F.softplus(beta_logit + beta_resid)
        return alpha, beta

    def adapter_residual(self, obs, privileged_query):
        _, _, alpha_resid, beta_resid = self._detached_teacher_parts(obs, privileged_query)
        return alpha_resid, beta_resid

    def adapter_parameters(self):
        return list(self.residual_adapter.parameters())

    def actor_parameters(self):
        return [
            *self.trunk.parameters(),
            *self.actor_alpha_head.parameters(),
            *self.actor_beta_head.parameters(),
        ]

    def critic_parameters(self):
        return [
            *self.critic_trunk.parameters(),
            *self.critic_head.parameters(),
        ]

    def act(self, obs):
        alpha, beta, value_logits = self.student_policy_and_value(obs)
        distribution = Beta(alpha, beta, validate_args=False)
        z = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        return action, z, value_logits

    def get_value(self, obs):
        feat = self._critic_feat(obs)
        return self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.actor_epochs > 0 and args.critic_epochs > 0
    assert args.adv_boost > 0.0, "a non-positive margin makes the teacher no better"
    assert args.env_id == "HalfCheetah-v4", "this versioned arm is HalfCheetah-v4 only"
    assert 0.95 <= args.jac_cos <= 1.0, "OPSD must retain at least 0.95 directional cosine"
    assert args.cond_lambda == 0.0, "privileged content must be the realized one-step TD residual"

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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
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
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, args).to(device)
    # Rollout forward only. The update stays eager: it is a small share of wall clock
    # (512 minibatch steps per iteration against 2048 acting steps), and graphing it would
    # complicate the dual/telemetry paths for little gain.
    act_fn = (
        torch.compile(agent.act, mode="reduce-overhead") if args.compile_act else agent.act
    )
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    adapter_params = agent.adapter_parameters()
    actor_ids = {id(p) for p in actor_params}
    critic_ids = {id(p) for p in critic_params}
    adapter_ids = {id(p) for p in adapter_params}
    all_ids = {id(p) for p in agent.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    assert actor_ids.isdisjoint(adapter_ids)
    assert critic_ids.isdisjoint(adapter_ids)
    assert actor_ids | critic_ids | adapter_ids == all_ids
    actor_optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)
    adapter_optimizer = optim.Adam(adapter_params, lr=args.learning_rate, eps=1e-5)
    adapter_generator = torch.Generator(device=device)
    adapter_generator.manual_seed(args.seed + 1_000_003)
    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    assert obs_shape is not None and action_shape is not None
    act_dim = int(np.prod(action_shape))
    # The transition head is omitted only for the algebraic identity control.
    use_jac = args.jac_cos < 1.0
    jac_model = None
    jac_optimizer = None
    if use_jac:
        with torch.random.fork_rng(devices=[device] if args.cuda else []):
            jac_model = TransitionHead(obs_shape[0], act_dim, args.jac_hidden).to(device)
        jac_optimizer = optim.Adam(jac_model.parameters(), lr=args.learning_rate, eps=1e-5)

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

    # Slow RMS of the raw conditioning residual. Not a gradient statistic -- it exists only
    # to keep the Fourier features inside the range their fixed frequencies resolve.
    cond_ms = torch.zeros((), device=device)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            critic_optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            adapter_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z, value_logits = act_fn(next_obs)
                values[step] = hl_support.to_scalar(value_logits[:, 0])
            latent_zs[step] = z

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
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

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

        # ================= GAE on V(s), then the MTP critic targets =====================
        with torch.no_grad():
            next_value_logits = agent.get_value(next_obses.reshape((-1,) + obs_shape))[:, 0]
            next_values = hl_support.to_scalar(next_value_logits).reshape(
                args.num_steps, args.num_envs
            )
            # Two credit signals from the SAME V(s). GAE(gae_lambda) for critic targets,
            # and GAE(cond_lambda) for the privileged channel.
            advantages = torch.zeros_like(rewards)
            cond_adv = torch.zeros_like(rewards)
            last_gae = torch.zeros(args.num_envs, device=device)
            last_cond = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_values[t] * bootstrap_nonterminal - values[t]
                last_gae = delta + args.gamma * args.gae_lambda * lambda_nonterminal * last_gae
                advantages[t] = last_gae
                last_cond = delta + args.gamma * args.cond_lambda * lambda_nonterminal * last_cond
                cond_adv[t] = last_cond
            returns = advantages + values

            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros((*returns.shape, mtp), dtype=torch.bool, device=device)
            for horizon in range(mtp):
                valid_len = args.num_steps - horizon
                valid_horizon = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=device
                )
                for boundary_offset in range(horizon):
                    valid_horizon &= (
                        transition_boundaries[boundary_offset : boundary_offset + valid_len] == 0
                    )
                return_mtp[:valid_len, :, horizon] = returns[horizon:]
                return_mtp_mask[:valid_len, :, horizon] = valid_horizon
            target_probs = hl_support.project(return_mtp)

            # THE PRIVILEGED CHANNEL must actually IDENTIFY the action, or the network is
            # right to ignore it. Measured with GAE(0.95): cond_gap 0.002 and falling,
            # distill_kl 3e-4 -- the teacher was the student. The cause is informational,
            # not architectural: with gamma 0.99 and lambda 0.95 the advantage is dominated
            # by ~100 steps of downstream trajectory and value error, so I(a_t ; A_t | s_t)
            # is almost nil. The 1-step residual delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
            # is instead a near-deterministic function of a_t in a deterministic MuJoCo
            # transition, so conditioning on it is learnable, and it is exactly the classic
            # actor-critic advantage -- V(s) only, no action ever enters the critic.
            b_adv = cond_adv.reshape(-1)
            # SCALING A CONDITIONING INPUT IS NOT SCALING A GRADIENT. "batch" (v1-v5,
            # inherited from PPO adv-norm) is wrong twice over here:
            #   (a) mean subtraction destroys delta's natural zero. delta_t = 0 means
            #       "exactly as V expected"; subtracting the batch mean relabels the
            #       least-bad action of an all-bad batch as POSITIVE -- a false sign.
            #   (b) the batch sd makes the units non-stationary, and adv_boost is quoted
            #       in those units, so the same nominal margin means different things at
            #       different times.
            # Raw delta cannot be fed either: AdvEmbed's frequencies are FIXED (0.5..8).
            # Measured (v4, 131k steps): raw delta's RMS grew 0.61 -> 2.17 in 4 iterations
            # and already saturated the +-3 clip at 11%. Entropy preservation ordered
            # raw > ema_rms > batch, exactly as (a) predicts.
            if args.cond_scale == "batch":
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            elif args.cond_scale == "ema_rms":
                ms = b_adv.square().mean()
                cond_ms.mul_(args.cond_ema_beta).add_((1.0 - args.cond_ema_beta) * ms)
                bias = 1.0 - args.cond_ema_beta ** iteration
                b_adv = b_adv / (cond_ms / bias).sqrt().clamp_min(1e-8)
            elif args.cond_scale == "raw":
                pass
            else:
                raise ValueError(f"unknown cond_scale {args.cond_scale!r}")
            cond_scale_used = b_adv.square().mean().sqrt().item()
            b_adv_cond = b_adv.clamp(-args.adv_cond_clip, args.adv_cond_clip).unsqueeze(-1)
            cond_clipped = (b_adv.abs() >= args.adv_cond_clip).float().mean().item()

        b_obs = obs.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).to(torch.float32)

        # ===== ADAPTER RATIONALIZATION, THEN A FROZEN RESIDUAL-OPSD TEACHER ===========
        if args.adv_boost_final >= 0.0:
            anneal_frac = (iteration - 1.0) / max(args.num_iterations - 1.0, 1.0)
            boost = args.adv_boost + anneal_frac * (args.adv_boost_final - args.adv_boost)
        else:
            boost = args.adv_boost
        query_all = (b_adv_cond + boost).clamp(-args.adv_cond_clip, args.adv_cond_clip)
        query_clip_frac = (
            (query_all.abs() >= args.adv_cond_clip - 1e-6).float().mean().item()
        )

        # Snapshot the rollout student before the adapter sees credit. The adapter cannot
        # change it, but taking the snapshot here makes the ownership boundary auditable.
        with torch.no_grad():
            old_alphas, old_betas = [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                a_s, b_s = agent.student_policy(b_obs[sl])
                old_alphas.append(a_s)
                old_betas.append(b_s)
            b_s_alpha_old = torch.cat(old_alphas).detach()
            b_s_beta_old = torch.cat(old_betas).detach()
            student_nll = -Beta(
                b_s_alpha_old, b_s_beta_old, validate_args=False
            ).log_prob(b_z).sum(-1).mean().item()

        # Exactly one fresh-data adapter pass. Its private Generator prevents this
        # rationalization permutation from advancing rollout/base-minibatch randomness.
        adapter_clone_losses, adapter_grad_norms = [], []
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        adapter_perm = torch.randperm(
            args.batch_size, generator=adapter_generator, device=device
        )
        for start in range(0, args.batch_size, args.minibatch_size):
            mb = adapter_perm[start : start + args.minibatch_size]
            a_cl, b_cl = agent.teacher_policy(b_obs[mb], b_adv_cond[mb])
            clone_loss = -Beta(a_cl, b_cl, validate_args=False).log_prob(
                b_z[mb]
            ).sum(-1).mean()
            adapter_optimizer.zero_grad(set_to_none=True)
            (args.clone_coef * clone_loss).backward()
            assert all(
                p.grad is None for p in [*actor_params, *critic_params]
            ), "clone gradient reached actor or critic"
            adapter_grad = nn.utils.clip_grad_norm_(adapter_params, args.max_grad_norm)
            adapter_optimizer.step()
            adapter_clone_losses.append(clone_loss.item())
            adapter_grad_norms.append(float(adapter_grad))
        adapter_optimizer.zero_grad(set_to_none=True)

        # Freeze both realized and optimistic teachers only after rationalization. All
        # subsequent actor gradients flow through the student and nowhere else.
        with torch.no_grad():
            realized_alphas, realized_betas = [], []
            query_alphas, query_betas = [], []
            residual_rms_chunks = []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                a_r, b_r = agent.teacher_policy(b_obs[sl], b_adv_cond[sl])
                a_q, b_q = agent.teacher_policy(b_obs[sl], query_all[sl])
                r_a, r_b = agent.adapter_residual(b_obs[sl], query_all[sl])
                realized_alphas.append(a_r)
                realized_betas.append(b_r)
                query_alphas.append(a_q)
                query_betas.append(b_q)
                residual_rms_chunks.append(
                    torch.cat([r_a, r_b], dim=-1).square().mean().sqrt().item()
                )
            b_real_alpha = torch.cat(realized_alphas).detach()
            b_real_beta = torch.cat(realized_betas).detach()
            b_unrot_alpha = torch.cat(query_alphas).detach()
            b_unrot_beta = torch.cat(query_betas).detach()
            b_t_alpha = b_unrot_alpha.clone()
            b_t_beta = b_unrot_beta.clone()
            adapter_residual_rms = float(np.mean(residual_rms_chunks))
            adapter_clone_nll = -Beta(
                b_real_alpha, b_real_beta, validate_args=False
            ).log_prob(b_z).sum(-1).mean().item()
            realized_teacher_student_kl = beta_kl_per_dim(
                b_real_alpha, b_real_beta, b_s_alpha_old, b_s_beta_old
            ).clamp_min(0.0).sum(-1).mean().item()
            query_teacher_student_kl = beta_kl_per_dim(
                b_unrot_alpha, b_unrot_beta, b_s_alpha_old, b_s_beta_old
            ).clamp_min(0.0).sum(-1).mean().item()
        assert all(
            not target.requires_grad and target.grad_fn is None
            for target in (b_t_alpha, b_t_beta, b_s_alpha_old, b_s_beta_old)
        )

        # ===== ROUTE 3: THE ANALYTIC IMPROVEMENT DIRECTION =============================
        # Everything else in this family builds its teacher from the action-credit
        # CORRELATION, which for an exponential-family policy is the policy gradient's own
        # information -- so it can sharpen the estimate but not change what is known. The
        # one information source that is different and still in charter is the pathwise
        # derivative through a learned one-step transition:
        #       d = d/dz [ gamma * V( s + ds_hat(s, a(z)) ) ]
        # V(s) stays the value object (no Q, no twin critics, no target nets), the model is
        # supervised regression on OBSERVED transitions of the single trajectory, and the
        # model is never rolled forward, so compounding error does not arise.
        #
        # VALIDATED OFFLINE BEFORE THIS FILE EXISTED (real HalfCheetah-v4 transitions,
        # learned J vs simulator central differences on 300 held-out states):
        #   n_train    held-out ds R2    cos(J^T e_vx)    cos(full J)
        #     5,000        0.841            0.955           0.969
        #   100,000        0.944            0.975           0.975
        # e_vx is obs index 8 = qvel[0] = forward velocity, i.e. exactly what HalfCheetah
        # pays for, so cos(J^T e_vx) IS the accuracy of the improvement direction. The
        # Jacobian magnitude is biased, so only its normalized direction is used; the
        # R2 gate interpolates the requested cosine back toward identity.
        jac_gate = 0.0
        jac_r2 = float("nan")
        jac_align = float("nan")
        jac_rot = float("nan")
        jac_dir_norm = float("nan")
        jac_clamp_frac = float("nan")
        jac_dose_scale = float("nan")
        jac_dose_rel_error = float("nan")
        b_next_obs = next_obses.reshape((-1,) + obs_shape)
        b_valid = (
            transition_valids * (1.0 - transition_terminations)
        ).reshape(-1)
        b_vsel = b_valid > 0
        if use_jac:
            assert jac_model is not None
            assert jac_optimizer is not None
            # `transition_valids` marks "a next observation exists", which is what GAE
            # needs. A one-step DYNAMICS model needs more: a hard termination is not an
            # ordinary physics step, and GAE itself already multiplies the two
            # (bootstrap_nonterminal = (1 - term) * valid). Use the same product here so the
            # model, its R2, and the direction all agree on what a transition is. No-op on
            # HalfCheetah-v4, which never terminates; correct on envs that do.
            with torch.no_grad():
                # HELD-OUT by construction: the head has only ever seen earlier rollouts.
                ds_true = b_next_obs - b_obs
                b_act = agent.action_low + (agent.action_high - agent.action_low) * b_z
                ds_pred = (
                    jac_model(b_obs, b_act)
                    * jac_model.ds_var.sqrt().clamp_min(1e-6)
                    + jac_model.ds_mean
                )
                w = b_valid.unsqueeze(-1)
                n_valid = float(b_valid.sum().item())
                if n_valid >= 2.0:
                    mu = (ds_true * w).sum(0) / n_valid
                    sse = (((ds_true - ds_pred) ** 2) * w).sum(0)
                    sst = (((ds_true - mu) ** 2) * w).sum(0).clamp_min(1e-12)
                    jac_r2 = float((1.0 - sse / sst).mean().item())
                # else: jac_r2 stays NaN. With no usable rows, sse and sst are both 0 and
                # the clamp would make R2 read exactly 1.0 -- a *perfect* model on this
                # file's primary safety signal, from zero data. NaN keeps the gate shut.
            # Guardrail, not an objective: an inaccurate model yields no rotation, so this
            # arm becomes the unrotated residual-OPSD arm for that iteration.
            jac_gate = float(
                min(max((jac_r2 - args.jac_r2_min) / (args.jac_r2_open - args.jac_r2_min), 0.0), 1.0)
            )
            if jac_gate > 0.0:
                dirs = []
                for start in range(0, args.batch_size, args.minibatch_size):
                    sl = slice(start, start + args.minibatch_size)
                    z_var = b_z[sl].detach().clone().requires_grad_(True)
                    act = (
                        agent.action_low
                        + (agent.action_high - agent.action_low) * z_var
                    )
                    s_next = jac_model.next_obs(b_obs[sl], act)
                    v_next = hl_support.to_scalar(agent.get_value(s_next)[:, 0])
                    (grad_z,) = torch.autograd.grad(
                        (args.gamma * v_next).sum(), z_var
                    )
                    dirs.append(grad_z.detach())
                jac_grad = torch.cat(dirs) * b_valid.unsqueeze(-1)
                with torch.no_grad():
                    conc = b_t_alpha + b_t_beta
                    mean_t = b_t_alpha / conc
                    sd_t = (mean_t * (1.0 - mean_t) / (conc + 1.0)).sqrt()
                    # The frozen rollout student names the displacement. The reference
                    # teacher here is this arm's unrotated residual-OPSD query, not the
                    # shared-conditional parent.
                    s_alpha_all = b_s_alpha_old
                    s_beta_all = b_s_beta_old
                    mean_s = s_alpha_all / (s_alpha_all + s_beta_all)
                    # Everything below lives in the policy's own Fisher/std coordinate:
                    # KL is locally quadratic in displacement/sd, and the pathwise
                    # covector's steepest-ascent direction in that metric is grad * sd.
                    u_cred = (mean_t - mean_s) / sd_t
                    g_u = jac_grad * sd_t
                    g_norm = g_u.norm(dim=-1, keepdim=True)
                    jac_dir_norm = float(g_norm[b_vsel].mean().item()) if b_vsel.any() else float("nan")
                    g_hat = g_u / g_norm.clamp_min(1e-8)
                    # EXACT ROTATION, not an emergent mix. The first version added
                    # `jac_step * g_hat` and renormalized. That is scale-dependent and it
                    # broke in practice: the same jac_step=0.10 that turned the teacher 57
                    # degrees at batch 1024 turned it only 21 degrees at batch 32768,
                    # because the achieved angle depends on ||u_cred||, which grows with the
                    # teacher's aggressiveness and drifts over training. A lever whose
                    # meaning moves with batch size and training stage cannot support a
                    # matched-step ladder.
                    #
                    # So specify the ANGLE directly. Decompose the analytic direction into
                    # the credit direction plus an orthogonal remainder, and rotate within
                    # that plane by exactly the requested cosine:
                    #     u_new = ||u_cred|| * (cos * u_hat + sin * g_perp)
                    # `jac_cos` is therefore the achieved rotation at every scale, while
                    # ||u_new|| == ||u_cred|| holds by construction.
                    u_norm = u_cred.norm(dim=-1, keepdim=True)
                    u_hat = u_cred / u_norm.clamp_min(1e-12)
                    g_perp = g_hat - (g_hat * u_hat).sum(-1, keepdim=True) * u_hat
                    perp_norm = g_perp.norm(dim=-1, keepdim=True)
                    g_perp = g_perp / perp_norm.clamp_min(1e-12)
                    cos_t = min(max(args.jac_cos, -1.0), 1.0)
                    cos_t = 1.0 - jac_gate * (1.0 - cos_t)  # gate closed => no rotation
                    sin_t = math.sqrt(max(1.0 - cos_t * cos_t, 0.0))
                    u_rot = u_norm * (cos_t * u_hat + sin_t * g_perp)
                    # Degenerate rows keep the unrotated residual-OPSD displacement exactly:
                    # a zero displacement has no plane, while a parallel/zero pathwise
                    # direction has no orthogonal component to rotate toward.
                    ok = (u_norm > 1e-9) & (perp_norm > 1e-9)
                    u_new = torch.where(ok, u_rot, u_cred)
                    # ===== MATCH THE DELIVERED BUDGET, NOT THE DISPLACEMENT NORM =========
                    # Holding ||u|| fixed is not sufficient because the objective clips
                    # each action-dimension KL. Rotation can change how often that clip
                    # binds. Match the batch's delivered clipped KL to the UNROTATED
                    # residual-OPSD teacher with one scalar along the rotated ray. The
                    # scalar may exceed one when rotation under-delivers; direction is
                    # unchanged, and the bisection still matches dose rather than norm.
                    def _delivered(u):
                        m = (mean_s + u * sd_t).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                        return (
                            beta_kl_per_dim(
                                m * conc, (1.0 - m) * conc, s_alpha_all, s_beta_all
                            )
                            .clamp_min(0.0)
                            .clamp(max=args.distill_kl_clip)
                            .sum(-1)
                            .mean()
                        )

                    unrotated_kl = _delivered(u_cred)
                    rotated_kl = _delivered(u_new)
                    lo = hi = 1.0
                    if rotated_kl > unrotated_kl:
                        lo, hi = 0.0, 1.0
                    elif rotated_kl < unrotated_kl:
                        lo, hi = 1.0, 2.0
                        for _ in range(args.jac_dose_iters):
                            if _delivered(hi * u_new) >= unrotated_kl:
                                break
                            lo, hi = hi, 2.0 * hi
                    if lo != hi:
                        for _ in range(args.jac_dose_iters):
                            mid = 0.5 * (lo + hi)
                            if _delivered(mid * u_new) > unrotated_kl:
                                hi = mid
                            else:
                                lo = mid
                    jac_dose_scale = 0.5 * (lo + hi)
                    u_new = jac_dose_scale * u_new
                    matched_kl = _delivered(u_new)
                    jac_dose_rel_error = float(
                        ((matched_kl - unrotated_kl).abs() / unrotated_kl.clamp_min(1e-12)).item()
                    )
                    mean_new = (mean_s + u_new * sd_t).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    # THE DECISIVE READOUT. If this cosine is ~1 the pathwise direction is
                    # just the credit direction rediscovered and route 3 buys nothing; if it
                    # is ~0 it is genuinely orthogonal information. Measured over valid
                    # rows in the metric where the mixing happens.
                    # On an invalid row g_hat is the zero
                    # vector, so cosine_similarity returns exactly 0 -- which would drag
                    # jac_align toward 0 and jac_rot toward 1, i.e. bias this arm's own
                    # pre-registered success criterion in the flattering direction by the
                    # invalid fraction. Small (~1-3% near episode boundaries) and entirely
                    # avoidable.
                    cos = torch.nn.functional.cosine_similarity
                    jac_align = float(cos(u_cred, g_hat, dim=-1)[b_vsel].mean().item())
                    # How far the frozen residual-OPSD teacher actually turned.
                    jac_rot = float(cos(u_cred, u_new, dim=-1)[b_vsel].mean().item())
                    # The clamp below is the one place the constant-dose invariant can
                    # plateau. It binds more often as the policy sharpens (measured ~0.004%
                    # of dims at parent-like doses, ~5% for a near-deterministic policy).
                    # Logging the binding fraction keeps the matched-dose ladder auditable
                    # instead of quietly drifting under budget at 8M.
                    jac_clamp_frac = float(
                        (
                            ((mean_s + u_new * sd_t) <= SAMPLE_EPS)
                            | ((mean_s + u_new * sd_t) >= 1.0 - SAMPLE_EPS)
                        )[b_vsel].float().mean().item()
                    )
                    b_t_alpha = mean_new * conc
                    b_t_beta = (1.0 - mean_new) * conc

        with torch.no_grad():
            teacher_entropy = Beta(
                b_t_alpha, b_t_beta, validate_args=False
            ).entropy().sum(-1).mean().item()
            teacher_mean = b_t_alpha / (b_t_alpha + b_t_beta)
            student_mean = b_s_alpha_old / (b_s_alpha_old + b_s_beta_old)
            cond_gap = (teacher_mean - student_mean).abs().mean().item()
        if use_jac:
            assert jac_model is not None
            # R2 and the teacher direction above used the previous rollout's internally
            # consistent (weights, stats) pair. Refresh statistics only after that held-out
            # evaluation, but before fitting on this rollout, so every standardized target
            # below is paired with the statistics that will decode its fitted head next time.
            with torch.no_grad():
                keep = b_vsel
                # ds.var(0) is the unbiased estimator; one row would yield NaN.
                if int(keep.sum().item()) >= 2:
                    jac_model.update_stats(
                        (b_next_obs[keep] - b_obs[keep]),
                        0.0 if iteration == 1 else 0.99,
                    )


        assert not b_t_alpha.requires_grad and not b_t_beta.requires_grad
        distill_kls, v_losses, actor_grad_norms, critic_grad_norms, jac_losses = [], [], [], [], []
        # Strict ownership is the intervention: the frozen-teacher KL and value regression
        # keep their original epoch/minibatch counts, but no parameter or optimizer state is
        # shared. Critic updates therefore cannot move the actor even through global clipping.
        for epoch in range(max(args.actor_epochs, args.critic_epochs)):
            do_actor = epoch < args.actor_epochs
            do_critic = epoch < args.critic_epochs
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]
                obs_mb = b_obs[mb]
                z_mb = b_z[mb]

                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                if do_actor:
                    a_tea, b_tea = b_t_alpha[mb], b_t_beta[mb]
                    assert not a_tea.requires_grad and not b_tea.requires_grad
                    a_stu, b_stu = agent.student_policy(obs_mb)
                    kl_dims = beta_kl_per_dim(a_tea, b_tea, a_stu, b_stu).clamp_min(0.0)
                    distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()
                    (args.distill_coef * distill_loss).backward()
                    assert all(p.grad is None for p in critic_params)
                    assert all(p.grad is None for p in adapter_params)
                    actor_grad_norms.append(grad_norm(actor_params))
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad(set_to_none=True)

                if do_critic:
                    value_logits = agent.get_value(obs_mb)
                    log_value_probs = torch.log_softmax(value_logits, dim=-1)
                    value_ce = -(b_target_probs[mb] * log_value_probs).sum(dim=-1)
                    v_loss = (value_ce * b_target_mask[mb]).sum(dim=-1).mean()
                    (args.vf_coef * v_loss).backward()
                    assert all(p.grad is None for p in actor_params)
                    assert all(p.grad is None for p in adapter_params)
                    critic_grad_norms.append(grad_norm(critic_params))
                    nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optimizer.step()
                    critic_optimizer.zero_grad(set_to_none=True)
                    v_losses.append(v_loss.item())

                if use_jac and do_actor:
                    assert jac_model is not None
                    assert jac_optimizer is not None
                    # Supervised only on observed transitions, with independent ownership.
                    ds_target = (
                        b_next_obs[mb] - obs_mb - jac_model.ds_mean
                    ) / jac_model.ds_var.sqrt().clamp_min(1e-6)
                    act_mb = agent.action_low + (agent.action_high - agent.action_low) * z_mb
                    vmask = b_valid[mb].unsqueeze(-1)
                    jac_loss = (
                        ((jac_model(obs_mb, act_mb) - ds_target) ** 2) * vmask
                    ).sum() / (vmask.sum().clamp_min(1.0) * obs_shape[0])
                    jac_optimizer.zero_grad(set_to_none=True)
                    (args.jac_coef * jac_loss).backward()
                    nn.utils.clip_grad_norm_(jac_model.parameters(), args.max_grad_norm)
                    jac_optimizer.step()
                    jac_losses.append(jac_loss.item())

        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        # Measure whether the updated student actually followed the frozen final teacher.
        with torch.no_grad():
            new_alphas, new_betas = [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                a_new, b_new = agent.student_policy(b_obs[sl])
                new_alphas.append(a_new)
                new_betas.append(b_new)
            b_s_alpha_new = torch.cat(new_alphas)
            b_s_beta_new = torch.cat(new_betas)
            old_student_new_student_kl = beta_kl_per_dim(
                b_s_alpha_old, b_s_beta_old, b_s_alpha_new, b_s_beta_new
            ).clamp_min(0.0).sum(-1).mean().item()
            teacher_kl_pre_rows = beta_kl_per_dim(
                b_t_alpha, b_t_beta, b_s_alpha_old, b_s_beta_old
            ).clamp_min(0.0).sum(-1)
            teacher_kl_post_rows = beta_kl_per_dim(
                b_t_alpha, b_t_beta, b_s_alpha_new, b_s_beta_new
            ).clamp_min(0.0).sum(-1)
            teacher_new_student_kl = teacher_kl_post_rows.mean().item()
            teacher_student_kl_reduction = (
                teacher_kl_pre_rows.mean() - teacher_kl_post_rows.mean()
            ).item()
            student_entropy = Beta(
                b_s_alpha_new, b_s_beta_new, validate_args=False
            ).entropy().sum(-1).mean().item()
            trunk_diff_sq = torch.zeros((), device=device)
            actor_trunk_sq = torch.zeros((), device=device)
            for actor_param, critic_param in zip(
                agent.trunk.parameters(), agent.critic_trunk.parameters(), strict=True
            ):
                trunk_diff_sq.add_((actor_param - critic_param).float().square().sum())
                actor_trunk_sq.add_(actor_param.float().square().sum())
            actor_critic_trunk_rel_distance = (
                trunk_diff_sq / actor_trunk_sq.clamp_min(1e-12)
            ).sqrt().item()


        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance = np.var(y_true)
        explained_variance = np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar(
            "charts/adapter_learning_rate",
            adapter_optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar(
            "charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "charts/critic_learning_rate", critic_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/adapter_clone_nll", adapter_clone_nll, global_step)
        writer.add_scalar(
            "losses/adapter_train_nll", float(np.mean(adapter_clone_losses)), global_step
        )
        writer.add_scalar("losses/student_nll", student_nll, global_step)
        writer.add_scalar(
            "debug/channel_nats",
            student_nll - adapter_clone_nll,
            global_step,
        )
        writer.add_scalar(
            "debug/realized_teacher_student_kl", realized_teacher_student_kl, global_step
        )
        writer.add_scalar(
            "debug/query_teacher_student_kl", query_teacher_student_kl, global_step
        )
        writer.add_scalar("debug/adapter_residual_rms", adapter_residual_rms, global_step)
        writer.add_scalar(
            "losses/adapter_grad_norm", float(np.mean(adapter_grad_norms)), global_step
        )
        writer.add_scalar(
            "losses/actor_grad_norm", float(np.mean(actor_grad_norms)), global_step
        )
        writer.add_scalar(
            "losses/critic_grad_norm", float(np.mean(critic_grad_norms)), global_step
        )
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/value_loss", float(np.mean(v_losses)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/student_entropy", student_entropy, global_step)
        writer.add_scalar(
            "debug/old_student_new_student_kl", old_student_new_student_kl, global_step
        )
        writer.add_scalar(
            "debug/teacher_new_student_kl", teacher_new_student_kl, global_step
        )
        writer.add_scalar(
            "debug/teacher_student_kl_reduction", teacher_student_kl_reduction, global_step
        )
        writer.add_scalar(
            "debug/actor_critic_trunk_rel_distance",
            actor_critic_trunk_rel_distance,
            global_step,
        )
        writer.add_scalar("debug/query_clip_frac", query_clip_frac, global_step)
        writer.add_scalar("debug/adv_boost", float(boost), global_step)
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/cond_scale_rms", cond_scale_used, global_step)
        writer.add_scalar("debug/cond_clip_frac", cond_clipped, global_step)
        writer.add_scalar("debug/cond_gap", cond_gap, global_step)
        writer.add_scalar("debug/teacher_entropy", teacher_entropy, global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        if use_jac:
            writer.add_scalar("debug/jac_r2", jac_r2, global_step)
            writer.add_scalar("debug/jac_gate", jac_gate, global_step)
            writer.add_scalar("debug/jac_dir_norm", jac_dir_norm, global_step)
            # THE decisive number: ~1 means route 3 rediscovered the credit direction and
            # buys nothing; ~0 means it is orthogonal information.
            writer.add_scalar("debug/jac_align", jac_align, global_step)
            writer.add_scalar("debug/jac_rot", jac_rot, global_step)
            writer.add_scalar("debug/jac_clamp_frac", jac_clamp_frac, global_step)
            # Scalar needed to preserve the unrotated teacher's delivered clipped-KL
            # budget. Values below/above one compensate over/under-delivery respectively.
            writer.add_scalar("debug/jac_dose_scale", jac_dose_scale, global_step)
            writer.add_scalar("debug/jac_dose_rel_error", jac_dose_rel_error, global_step)
            writer.add_scalar(
                "losses/jac_model", float(np.mean(jac_losses)) if jac_losses else float("nan"),
                global_step,
            )
        print("SPS:", sps)

    envs.close()
    writer.close()
