# OPSD-AdvCond v1 -- on-policy self-distillation with the realized advantage as the
# privileged context. One network, two contexts. No PPO anywhere.
# =====================================================================================
# THE PAPER (arXiv 2601.18734, On-Policy Self-Distillation). One LLM instantiates both
# policies under different conditioning: p_S(.|x) sees only the prompt, p_T(.|x,y*) also
# sees the ground-truth solution y*. The student rolls out, both policies re-read the
# student's own tokens, and the loss is a per-token distribution divergence D(p_T||p_S)
# with gradients flowing ONLY through the student. The teacher is not a bigger network,
# it is the same weights handed privileged information it can rationalize.
#
# THE RL TRANSLATION USED HERE
#   "token"            -> one action dimension (per-dim divergence, per-dim clipping)
#   "privileged y*"    -> THE REALIZED ADVANTAGE of the action actually taken, which the
#                         acting policy provably could not see at action time
#   "rationalization"  -> supervised conditional density fit p(a | s, A)
#   "student context"  -> the same input with the privileged slot ZEROED
#
# So: roll out, record what it did and how well it scored, then go over every step again.
# In the teacher context the network is told the advantage its own action earned and is
# fit to that action; in the student context that slot is zero. Querying the same weights
# at an OPTIMISTIC advantage asks "what would I do if this were a k-sigma better action?"
# and that answer is distilled back into the unconditioned policy. It bootstraps itself.
#
# WHY THE ACTION IS NOT AN INPUT (the one non-obvious design constraint)
# It is tempting to feed (s, a, A) since the model "sees what it did". That is degenerate:
# the rationalization loss is -log p(a | s, ...), so if a appears on BOTH sides the fit
# collapses to the identity map (a Dirac at a), driving the loss to -inf while teaching
# the network nothing about improvement. The teacher would then be exactly the rollout
# policy and the distillation term would be identically zero. So "what it did" enters as
# the regression TARGET and "how well it scored" enters as the conditioning INPUT.
#
# TWO OBJECTIVES (no ratio, no clipped surrogate, no policy gradient)
#   1. Rationalization:  L_clone = -log pi(a_t | s_t, A_t)      [teacher context]
#      A dense, low-variance supervised regression. It makes the privileged channel mean
#      something, and it anchors the conditional to on-policy data, which doubles as a
#      natural trust region and resists entropy collapse.
#   2. Self-distillation: L_distill = sum_d min( KL(pi_T,d || pi_S,d), tau )
#      pi_T = pi(.|s, A=+adv_query) DETACHED, pi_S = pi(.|s, 0). Gradients reach only the
#      student context. This is the paper's per-token clipped divergence, verbatim in form.
#
# CRITIC. Stays V(s): no action input, evaluated ONLY with the privileged slot zeroed.
# The advantage is derived from V, so letting V see it would be circular; zeroing keeps
# the value head an honest state-value function and keeps its bias low. HL-Gauss
# multi-token-prediction critic and GAE are inherited unchanged from the 8278@8M chassis.
#
# HYPOTHESIS. The improvement operator is a conditional-density query rather than a
# gradient ascent step, so it inherits the variance of supervised learning, not of policy
# gradients, while remaining strictly on-policy. Advantage conditioning is a scalar, so
# the fit is well posed even when the per-dim advantage/action correlation is weak.
#
# KILL-TELLS (all logged)
#   debug/cond_gap ~ 0        -> the trunk is ignoring the privileged slot; the teacher is
#                                the student, the method is inert. Raise adv_query or the
#                                clone weight.
#   losses/distill_kl ~ 0 with flat returns -> same failure, seen from the loss side.
#   entropy collapsing while cond_gap grows -> adv_query is extrapolating off the fit.
# =====================================================================================
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
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4

    # --- OPSD advantage conditioning ---
    adv_query: float = 1.0        # optimistic advantage the teacher is queried at, in sigma
    adv_cond_clip: float = 3.0    # clamp on the standardized advantage used as conditioning
    adv_embed_freqs: int = 8      # sinusoidal features per phase; privileged block is 2x this
    cond_lambda: float = 0.0      # GAE lambda for the PRIVILEGED channel only; 0 = 1-step
                                  # TD residual, the most action-attributable credit signal
    clone_coef: float = 1.0       # weight on the rationalization fit p(a | s, A)
    distill_coef: float = 1.0     # weight on the per-dim teacher->student divergence
    distill_kl_clip: float = 2.0  # tau: the paper's per-token pointwise divergence clip

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


class Agent(nn.Module):
    """One network, two contexts. The privileged block is the TRAILING input channels.

    Present  -> Fourier features of the realized advantage (teacher context).
    Absent   -> that whole block is zeroed (student context, and always for the critic).
    Zeroing rather than feeding A=0 keeps "no privileged information" a distinct code,
    since cos(0)=1 means the embedding of zero is not the zero vector.
    """

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.adv_embed = AdvEmbed(args.adv_embed_freqs)
        self.cond_dim = self.adv_embed.dim
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

    def _feat(self, obs, cond):
        return self.trunk(torch.cat([obs, cond], dim=-1))

    def cond_present(self, adv):
        """Privileged context: Fourier features of the standardized advantage."""
        return self.adv_embed(adv)

    def _zero_cond(self, obs):
        """Privileged context ABSENT: the whole block is zero."""
        return obs.new_zeros((obs.shape[0], self.cond_dim))

    def policy(self, obs, cond):
        feat = self._feat(obs, cond)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        return alpha, beta

    def policy_and_value(self, obs, cond):
        feat = self._feat(obs, cond)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(feat))
        value_logits = self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return alpha, beta, value_logits

    def act(self, obs):
        """Acting policy == the STUDENT context (privileged slot zeroed)."""
        alpha, beta, value_logits = self.policy_and_value(obs, self._zero_cond(obs))
        distribution = Beta(alpha, beta, validate_args=False)
        z = distribution.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        return action, z, value_logits

    def get_value(self, obs):
        """V(s): no action, privileged slot always zeroed."""
        feat = self._feat(obs, self._zero_cond(obs))
        return self.critic_head(feat).view(-1, self.critic_mtp_horizon, self.num_bins)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    assert args.batch_size % args.num_minibatches == 0
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.update_epochs > 0
    assert args.adv_query <= args.adv_cond_clip, "teacher must be queried inside the fit"

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

    envs = gym.vector.SyncVectorEnv(
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
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

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                action, z, value_logits = agent.act(next_obs)
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
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            b_adv_cond = b_adv.clamp(-args.adv_cond_clip, args.adv_cond_clip).unsqueeze(-1)

        b_obs = obs.reshape((-1,) + obs_shape)
        b_z = latent_zs.reshape(-1, act_dim)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).to(torch.float32)

        # ============ replay every step: rationalize, then self-distill ==================
        clone_losses, distill_kls, v_losses, ents, gaps, tea_ents = [], [], [], [], [], []
        for _ in range(args.update_epochs):
            permutation = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = permutation[start : start + args.minibatch_size]
                obs_mb, z_mb, adv_mb = b_obs[mb], b_z[mb], b_adv_cond[mb]
                n = obs_mb.shape[0]

                # Teacher: same weights, queried at an optimistic advantage. Detached, so
                # gradients reach only the student -- exactly the paper's asymmetry.
                cond_query = agent.cond_present(torch.full_like(adv_mb, args.adv_query))
                with torch.no_grad():
                    a_tea, b_tea = agent.policy(obs_mb, cond_query)

                # One forward for both remaining contexts:
                #   [privileged at the REALIZED A_t ; privileged ABSENT (the student)].
                cond_absent = adv_mb.new_zeros((n, agent.cond_dim))
                alpha, beta, value_logits = agent.policy_and_value(
                    torch.cat([obs_mb, obs_mb], 0),
                    torch.cat([agent.cond_present(adv_mb), cond_absent], 0),
                )
                a_cl, b_cl = alpha[:n], beta[:n]
                a_stu, b_stu = alpha[n:], beta[n:]

                # 1. Rationalization: fit p(a_t | s_t, A_t). "What it did" is the target.
                clone_loss = -(
                    Beta(a_cl, b_cl, validate_args=False).log_prob(z_mb).sum(-1).mean()
                )

                # 2. Per-dim clipped forward KL from the detached teacher into the student.
                kl_dims = beta_kl_per_dim(a_tea, b_tea, a_stu, b_stu).clamp_min(0.0)
                distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()

                # 3. Critic: V(s) from the zeroed-context half only.
                log_value_probs = torch.log_softmax(value_logits[n:], dim=-1)
                value_ce = -(b_target_probs[mb] * log_value_probs).sum(dim=-1)
                v_loss = (value_ce * b_target_mask[mb]).sum(dim=-1).mean()

                loss = (
                    args.clone_coef * clone_loss
                    + args.distill_coef * distill_loss
                    + args.vf_coef * v_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clone_losses.append(clone_loss.item())
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    v_losses.append(v_loss.item())
                    ents.append(
                        Beta(a_stu, b_stu, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    tea_ents.append(
                        Beta(a_tea, b_tea, validate_args=False).entropy().sum(-1).mean().item()
                    )
                    # Does the trunk actually USE the privileged slot? If this is 0 the
                    # teacher is the student and the whole method is a no-op.
                    gaps.append(
                        (
                            a_tea / (a_tea + b_tea) - a_stu / (a_stu + b_stu)
                        ).abs().mean().item()
                    )

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance = np.var(y_true)
        explained_variance = np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/clone_nll", float(np.mean(clone_losses)), global_step)
        writer.add_scalar("losses/distill_kl", float(np.mean(distill_kls)), global_step)
        writer.add_scalar("losses/value_loss", float(np.mean(v_losses)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("debug/cond_gap", float(np.mean(gaps)), global_step)
        writer.add_scalar("debug/teacher_entropy", float(np.mean(tea_ents)), global_step)
        writer.add_scalar("debug/student_entropy", float(np.mean(ents)), global_step)
        writer.add_scalar("debug/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        print("SPS:", sps)

    envs.close()
    writer.close()
