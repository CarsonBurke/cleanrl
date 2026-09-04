# HOPSD v6 "cohex" — coherent exploration makes hindsight informative.
# =====================================================================================
# v6 = v2 (awr05/kl005 config) + a per-env AR(1) exploration latent u_t injected into
# the Beta heads. THE DIAGNOSIS THAT FORCES THIS: in every v2-family run the teacher's
# conditional entropy given (s, phi) EQUALS the student's marginal entropy given s
# (-7.32 vs -7.31 in awr05_r2) — the privileged future carries ZERO incremental
# information about a_t. Structural cause: with iid per-step exploration noise, future
# actions are independent of a_t's noise component given s_t; hindsight can only
# recover the policy mean, which s_t already determines. All improvement so far has
# come from the AWR tilt alone; the OPSD channel was informationally dead.
#
# Fix: make the exploration legible to hindsight. u_{t+1} = rho*u_t + sqrt(1-rho^2)*eps
# (rho=0.95 ~ the H=20 window; fresh draw per episode), injected with fixed gain:
#   alpha = 1 + softplus(h_a(feat) + g*u),  beta = 1 + softplus(h_b(feat) - g*u)
# so each episode runs a coherent persistent "experiment" (u>0 biases that action dim
# up for ~20 steps). Consequences:
#   - a_{t+1..t+H} now carry u's signature -> phi genuinely informs a_t's exploration
#     component -> the teacher conditional can sharpen BELOW the student marginal
#     (log debug/hindsight_info_gap = teacher_ent - student_ent; negative = alive).
#   - The AWR tilt selects among coherent experiments, not per-step dither, and the
#     distilled correction is a whole-gait nudge.
#   - u is part of the Markov state: actor trunk AND critic see [obs, u]; GAE
#     bootstraps at the natural AR continuation of u across truncations.
#   - The teacher does NOT see u: it must infer the experiment from its consequences
#     (future actions). Feeding u would pin a_t and kill the tilt (v1's lesson).
# =====================================================================================
#
# --- v1 method (retained below) ---
# Port of On-Policy Self-Distillation onto the iterthink_v24_beta_d3bucket_mtp base
# (config of the ppoadvnorm_batch_v1 run: raw GAE, batch-scope z-scoring). OPSD's LLM
# recipe: a TEACHER conditioned on privileged info (the verified solution) and a STUDENT
# conditioned on the problem only; the student rolls out; at every position the loss is
# per-position forward KL(teacher || student) over the FULL distribution, with pointwise
# per-entry clipping min(l, tau); gradients flow only through the student. Dense
# distillation replaces sparse-reward RL.
#
# RL translation (no verified answer exists, so hindsight is the privilege):
#   TEACHER ACTOR  pi_T(a_t | s_t, phi_t): a separate ThinkTrunk that sees the realized
#     future phi_t over the next H=20 steps (~ GAE effective horizon 1/(1-gamma*lambda)):
#     future-action mean/std per dim (a_{t+1..t+H}; a_t excluded so the teacher cannot be
#     an identity map), the z-scored lambda-return g_t ("returns for the horizon"), and
#     the valid-horizon fraction. Trained by ADVANTAGE-WEIGHTED Beta NLL on the taken
#     native z: w = exp(adv_z/awr_temp).clamp(w_max), mean-normalized. The weighting is
#     the improvement operator (AWR): given hindsight, the teacher fits "the action that
#     should have been taken", not merely the action that was taken. Rationalization
#     (fit a_t given what followed) is far easier than generation — the paper's core bet.
#   TEACHER CRITIC V_T(s_t, gait_t): a separate ThinkTrunk over [obs, future-action
#     mean/std, valid frac] (NO return features — with them it degenerates into a return
#     copier). HL-Gauss CE to lambda-returns; a horizon-Q "V(s, plan)". v1 role:
#     diagnostic (its EV measures how informative the privileged plan summary is).
#   STUDENT: the unchanged base agent (shared ThinkTrunk -> Beta actor + Dreamer3-bucket
#     511-bin HL-Gauss MTP critic). Its actor objective is ONLY the dense distillation
#     loss sum_d min(KL(Beta_T,d || Beta_S,d), tau) at every rollout state, teacher
#     detached — PPO (ratio/clip/advantages in the actor loss) is fully removed. The
#     critic keeps its raw-return CE loss (it anchors GAE -> adv_z, g). target_kl stays
#     as a drift leash (epoch early-stop off replayed-z logprobs), not as an objective.
#
# Faithful-to-paper choices: forward KL (their decisive winner over reverse KL/JSD);
# full-distribution matching (closed-form Beta KL = the "full-vocab logit" analog);
# pointwise per-dim clipping (their per-vocab-entry min(l, tau)); teacher evaluated at
# the ACHIEVED hindsight context on every student state (no elites, no relabeling);
# gradients only through the student. Deviations forced by RL-from-scratch: the teacher
# must learn online (no frozen-at-init teacher) and needs the AWR weighting because a
# rollout future, unlike a verified solution, is not necessarily good.
#
# HYPOTHESIS: hindsight rationalization + advantage weighting make the teacher a
# per-state full-distribution target that is denser and lower-variance than PPO's
# clipped surrogate, so the student improves at every state including bad ones.
# Falsifiable: if the teacher is just posterior reconstruction (no improvement), the
# student converges to self-BC and returns plateau far below the PPO baseline
# (ppoadvnorm_batch_v1: 5012@2M / 6716@4M / 8278@8M).
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
    update_epochs: int = 10
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # student drift leash (epoch early-stop; no PPO objective)

    # v21 machinery kept from the base: shared student backbone + decoupled clipping.
    share_backbone: bool = True
    separate_grad_clip: bool = True
    actor_grad_clip: float = 0.25    # max-norm for the distill gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Dreamer3-style bucket critic (student MTP + teacher single-horizon share the support).
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

    # --- HOPSD ---
    hindsight_horizon: int = 20      # H: future window for the privileged features (~1/(1-gamma*lambda))
    noise_rho: float = 0.95          # AR(1) coefficient of the exploration latent (~20-step memory)
    noise_gain: float = 0.5          # fixed injection gain of u into the Beta heads
    awr_temp: float = 0.5            # teacher NLL weight temperature on batch-z-scored GAE
    awr_weight_max: float = 20.0     # clamp on exp(adv_z/temp) before mean-normalization
    distill_coef: float = 1.0        # student actor loss = distill_coef * clipped forward KL
    distill_kl_clip: float = 2.0     # tau: pointwise per-action-dim KL clip (paper's min(l, tau))
    teacher_conc_cap: float = 100.0  # hard cap on teacher Beta concentrations (sane sharp targets)
    teacher_vf_coef: float = 0.5     # teacher critic CE weight inside the teacher update
    teacher_grad_clip: float = 0.5   # teacher's own global clip (separate optimizer)
    teacher_sees_g: bool = False     # v2: g in the teacher-actor context kills the AWR tilt (v1 fixed point)

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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

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
    """Student: unchanged base agent (Beta actor + HL-Gauss MTP critic on a shared trunk)."""

    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.noise_gain = args.noise_gain
        self.share_backbone = args.share_backbone
        in_dim = obs_dim + act_dim  # [obs, u]: the exploration latent is Markov state
        if self.share_backbone:
            self.trunk = ThinkTrunk(in_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(in_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(in_dim, H, args.k_blocks, args.n_experts)
        self.num_bins = args.num_bins
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * args.num_bins, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )

    def _trunks(self, x):
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x, u):
        _, critic_feat = self._trunks(torch.cat([x, u], dim=-1))
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, u, z=None):
        # z is the native Beta sample in (0,1); replaying it recomputes log_prob at the
        # same sample (the base's z-replay). Also returns the Beta params for distillation.
        # u shifts the heads with fixed gain: coherent exploration exists from step 0,
        # independent of what the trunk has learned about u.
        actor_feat, critic_feat = self._trunks(torch.cat([x, u], dim=-1))
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat) + self.noise_gain * u)
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat) - self.noise_gain * u)
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self.action_low + (self.action_high - self.action_low) * z
        log_prob = dist.log_prob(z).sum(1)  # constant rescale Jacobian dropped (cancels)
        entropy = dist.entropy().sum(1)
        value_logits = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)
        return action, z, log_prob, entropy, value_logits, alpha, beta

    def actor_parameters(self):
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


class HindsightTeacher(nn.Module):
    """Separate teacher actor + teacher critic, both privileged.

    Actor input:  [obs, future-action mean/std per dim, g (z-scored lambda-return), valid frac]
    Critic input: [obs, future-action mean/std per dim, valid frac]  (no return features)
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        actor_in = obs_dim + 2 * act_dim + 1 + (1 if args.teacher_sees_g else 0)
        critic_in = obs_dim + 2 * act_dim + 1
        self.actor_trunk = ThinkTrunk(actor_in, H, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.critic_trunk = ThinkTrunk(critic_in, H, args.k_blocks, args.n_experts)
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.conc_cap = args.teacher_conc_cap

    def actor_params_for(self, x_priv):
        feat = self.actor_trunk(x_priv)
        alpha = (1.0 + F.softplus(self.alpha_head(feat))).clamp(max=self.conc_cap)
        beta = (1.0 + F.softplus(self.beta_head(feat))).clamp(max=self.conc_cap)
        return alpha, beta

    def critic_logits(self, x_gait):
        return self.critic_head(self.critic_trunk(x_gait))


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


def build_future_features(actions, boundaries, horizon):
    """Per-(t, env) mean/std of a_{t+1..t+H} and the valid-horizon fraction.

    actions: (T, B, A); boundaries: (T, B) 1.0 where the transition at t ends an episode.
    Future step t+k (k>=1) is valid iff t+k <= T-1 and no boundary in transitions t..t+k-1.
    Returns (mean, std, valid_frac) with zeros where no valid future exists.
    """
    T, B, A = actions.shape
    valid = torch.ones(T, B, device=actions.device)
    s1 = torch.zeros(T, B, A, device=actions.device)
    s2 = torch.zeros(T, B, A, device=actions.device)
    cnt = torch.zeros(T, B, device=actions.device)
    for k in range(1, horizon + 1):
        if k > T - 1:
            break
        # extending the window by one step requires transition t+k-1 to be non-boundary
        valid = valid.clone()
        valid[: T - k] = valid[: T - k] * (1.0 - boundaries[k - 1 : T - 1])
        valid[T - k :] = 0.0  # window would run past the rollout
        m = valid.unsqueeze(-1)
        a_k = torch.zeros_like(actions)
        a_k[: T - k] = actions[k:]
        s1 = s1 + m * a_k
        s2 = s2 + m * a_k.pow(2)
        cnt = cnt + valid
    denom = cnt.clamp_min(1.0).unsqueeze(-1)
    mean = s1 / denom
    var = (s2 / denom - mean.pow(2)).clamp_min(0.0)
    std = var.sqrt()
    has_future = (cnt > 0).float().unsqueeze(-1)
    mean = mean * has_future
    std = std * has_future
    valid_frac = cnt / float(horizon)
    return mean, std, valid_frac


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

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
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
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    teacher = HindsightTeacher(obs_dim, act_dim, args).to(device)
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=args.learning_rate, eps=1e-5)

    hl_support = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, device
    )
    hl_support_cpu = Dreamer3BucketHLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.value_sigma_to_bin_ratio, torch.device("cpu")
    )

    def value_logits_to_scalar(logits):
        return hl_support.to_scalar(logits)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    us = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    u_boots = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    u_cur = torch.randn(args.num_envs, act_dim, device=device)  # stationary AR(1) start

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            us[step] = u_cur
            with torch.no_grad():
                action, z, logprob, ent, value_logits, _, _ = agent.get_action_and_value(next_obs, u_cur)
                values[step] = value_logits_to_scalar(value_logits[:, 0])
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
            # AR(1) evolution. Bootstrap uses the natural continuation (the u that WOULD
            # have applied at the next step of this episode); the live latent resets to a
            # fresh stationary draw on episode boundaries (each episode = one experiment).
            u_nat = args.noise_rho * u_cur + (1.0 - args.noise_rho**2) ** 0.5 * torch.randn_like(u_cur)
            u_boots[step] = u_nat
            u_cur = torch.where(
                next_done.unsqueeze(-1) > 0, torch.randn_like(u_cur), u_nat
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_transition_value_logits = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape),
                u_boots.reshape(-1, act_dim),
            )[:, 0]
            next_transition_values = value_logits_to_scalar(next_transition_value_logits).reshape(
                args.num_steps, args.num_envs
            )
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

            # Student-critic MTP targets (unchanged from the base).
            mtp = args.critic_mtp_horizon
            return_mtp = returns.new_zeros((*returns.shape, mtp))
            return_mtp_mask = torch.zeros(
                (*returns.shape, mtp), dtype=torch.bool, device=returns.device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones(
                    (valid_len, args.num_envs), dtype=torch.bool, device=returns.device
                )
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                return_mtp[:valid_len, :, h] = returns[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            target_probs = hl_support_cpu.project(return_mtp.detach().cpu())

            # --- HOPSD privileged context ---
            fut_mean, fut_std, fut_valid_frac = build_future_features(
                actions, transition_boundaries, args.hindsight_horizon
            )
            adv_z = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # batch scope
            g = (returns - returns.mean()) / (returns.std() + 1e-8)                # batch scope
            awr_w = (adv_z / args.awr_temp).exp().clamp(max=args.awr_weight_max)
            awr_w = awr_w / awr_w.mean()
            actor_ctx = [obs, fut_mean, fut_std]
            if args.teacher_sees_g:
                actor_ctx.append(g.unsqueeze(-1))
            actor_ctx.append(fut_valid_frac.unsqueeze(-1))
            teacher_actor_in = torch.cat(actor_ctx, dim=-1)
            teacher_critic_in = torch.cat(
                [obs, fut_mean, fut_std, fut_valid_frac.unsqueeze(-1)], dim=-1
            )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_us = us.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_teacher_actor_in = teacher_actor_in.reshape(-1, teacher_actor_in.shape[-1])
        b_teacher_critic_in = teacher_critic_in.reshape(-1, teacher_critic_in.shape[-1])
        b_awr_w = awr_w.reshape(-1)

        b_inds = np.arange(args.batch_size)
        distill_kls, distill_clipfracs, teacher_nlls = [], [], []
        # The target_kl leash freezes only the STUDENT; the teacher always trains its
        # full epochs (weighted MLE is off-policy safe, and a lagging teacher is worst
        # exactly when distillation drifts the student fast).
        student_stopped = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # ---- teacher update (own optimizer; nothing here touches the student) ----
                t_alpha, t_beta = teacher.actor_params_for(b_teacher_actor_in[mb_inds])
                t_dist = Beta(t_alpha, t_beta)
                t_nll = -t_dist.log_prob(b_latent_zs[mb_inds]).sum(-1)
                teacher_actor_loss = (b_awr_w[mb_inds] * t_nll).mean()
                t_value_logits = teacher.critic_logits(b_teacher_critic_in[mb_inds])
                t_target = b_target_probs[mb_inds, 0].to(device=device, non_blocking=True)
                t_v_loss = -(t_target * torch.log_softmax(t_value_logits, dim=-1)).sum(-1).mean()
                teacher_loss = teacher_actor_loss + args.teacher_vf_coef * t_v_loss
                teacher_optimizer.zero_grad(set_to_none=True)
                teacher_loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), args.teacher_grad_clip)
                teacher_optimizer.step()
                teacher_nlls.append(teacher_actor_loss.item())

                if student_stopped:
                    continue

                # ---- student update: dense clipped forward-KL distillation + critic CE ----
                _, _, newlogprob, entropy, value_logits, s_alpha, s_beta = agent.get_action_and_value(
                    b_obs[mb_inds], b_us[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                kl_dims = beta_kl_per_dim(t_alpha.detach(), t_beta.detach(), s_alpha, s_beta)
                kl_dims = kl_dims.clamp_min(0.0)
                distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()
                with torch.no_grad():
                    distill_kls.append(kl_dims.sum(-1).mean().item())
                    distill_clipfracs.append((kl_dims > args.distill_kl_clip).float().mean().item())
                pg_loss = args.distill_coef * distill_loss

                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                target_probs_mb = b_target_probs[mb_inds].to(device=value_logits.device, non_blocking=True)
                value_ce = -(target_probs_mb * value_log_probs).sum(dim=-1)
                value_mask = b_target_mask[mb_inds].to(
                    device=value_logits.device, dtype=value_ce.dtype, non_blocking=True
                )
                v_loss = (value_ce * value_mask).sum(dim=-1).mean()

                entropy_loss = entropy.mean()

                if args.separate_grad_clip:
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - args.ent_coef * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    for p, grad in value_grads:
                        p.grad = grad if p.grad is None else p.grad + grad
                    optimizer.step()
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if (
                args.target_kl is not None
                and not student_stopped
                and approx_kl > args.target_kl
            ):
                student_stopped = True

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Full-batch teacher diagnostics (chunked, no grad): privileged-critic EV,
        # teacher/student entropies, mean-action gap in native z space.
        with torch.no_grad():
            t_vals, t_ents, s_ents, gaps = [], [], [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                ta, tb = teacher.actor_params_for(b_teacher_actor_in[sl])
                t_ents.append(Beta(ta, tb).entropy().sum(-1).mean().item())
                _, _, _, s_ent, _, sa, sb = agent.get_action_and_value(b_obs[sl], b_us[sl], b_latent_zs[sl])
                s_ents.append(s_ent.mean().item())
                gaps.append((ta / (ta + tb) - sa / (sa + sb)).abs().mean().item())
                t_vals.append(hl_support.to_scalar(teacher.critic_logits(b_teacher_critic_in[sl])))
            t_vals = torch.cat(t_vals).cpu().numpy()
            teacher_ev = np.nan if var_y == 0 else 1 - np.var(y_true - t_vals) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("losses/teacher_nll", np.mean(teacher_nlls), global_step)
        writer.add_scalar("losses/teacher_value_loss", t_v_loss.item(), global_step)
        writer.add_scalar("losses/distill_kl", np.mean(distill_kls), global_step)
        writer.add_scalar("debug/distill_clipfrac", np.mean(distill_clipfracs), global_step)
        writer.add_scalar("debug/teacher_ev", teacher_ev, global_step)
        writer.add_scalar("debug/teacher_entropy", np.mean(t_ents), global_step)
        writer.add_scalar("debug/student_entropy", np.mean(s_ents), global_step)
        writer.add_scalar("debug/teacher_student_mean_gap", np.mean(gaps), global_step)
        writer.add_scalar("debug/hindsight_info_gap", np.mean(t_ents) - np.mean(s_ents), global_step)
        writer.add_scalar("debug/awr_weight_max", awr_w.max().item(), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/fut_valid_frac", fut_valid_frac.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
