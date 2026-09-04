# HOPSD v4 "plangrad" — hindsight plan-gradient distillation.
# =====================================================================================
# v4 replaces v2/v3's AWR reweighting with a FIRST-ORDER COUNTERFACTUAL improvement
# operator. Diagnosis of v2/v3 (adversarial review, confirmed by teacher_nll tracking
# entropy): each (state, future) pair is unique, so weighted MLE's optimum is plain
# reconstruction of the taken action — the advantage tilt lives only in underfitting
# and decays over the 10 epochs. The paper's privilege (y*) is verified-good; a rollout
# future is not, and reweighting is too weak a substitute.
#
# v4 inverts the division of labor so reconstruction stops being the failure mode and
# becomes the requirement:
#   DECODER  pi_T(a_t | s_t, phi_t): separate ThinkTrunk over [obs, phi, valid_frac],
#     phi = [future-action mean/std over H=20]. Trained by PLAIN (unweighted) Beta NLL
#     on the taken z — pure posterior reconstruction: "what action produces this plan".
#   PLAN CRITIC V_T(s_t, phi_t): separate ThinkTrunk, HL-Gauss CE to lambda-returns
#     (EV ~0.92 in v2, where it was only a diagnostic; here it is load-bearing).
#   IMPROVEMENT: per state, ascend the plan itself — phi* = phi + plan_steps normalized
#     gradient steps on V_T(s, phi), MEAN DIMS ONLY (the std dims' dV is confounded:
#     noise hurts return, so ascent would shrink them and arm an entropy ratchet through
#     the decoder), in a SIGMA-SCALED trust region (lengths in units of the batch per-dim
#     plan spread, so the counterfactual shrinks as the policy concentrates), clamped to
#     the valid plan box, reverted per-sample where V_T(phi*) < V_T(phi). "How the gait
#     SHOULD have differed" — a dense counterfactual, not a filter on what was done.
#   STUDENT: unchanged base agent; actor loss is ONLY the per-dim clipped forward KL
#     toward the decoder evaluated at the IMPROVED plan, pi_T(. | s, phi*), detached.
#     Critic keeps its HL-Gauss MTP CE (anchors GAE). No PPO anywhere in the student.
#
# Cold start is safe: the plan-critic head is zero-init -> V flat -> zero plan gradient
# -> phi* = phi -> distillation reduces to hindsight BC until the critic learns.
# target_kl=0.05 freezes the student only (decoder+critic always train full epochs).
#
# HYPOTHESIS: critic-gradient plan improvement is a stronger, non-decaying improvement
# operator than AWR reweighting, and plan-space steps give temporally coherent policy
# improvement (a whole-gait nudge, not per-step noise). Falsifiable: if grad(V_T) wrt
# phi carries no signal beyond obs, plan_dv ~ 0 and v4 stalls at self-BC; if the critic
# is exploited OOD, plan_dv is large but returns lag kl005_r2 (5162@2M baseline pace).
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
    target_kl: float = 0.05          # student drift leash (epoch early-stop; no PPO objective)

    share_backbone: bool = True
    separate_grad_clip: bool = True
    actor_grad_clip: float = 0.25    # max-norm for the distill gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Dreamer3-style bucket critic (student MTP + plan critic share the support).
    num_bins: int = 511
    v_min: float = -9.90353755128617
    v_max: float = 9.90353755128617
    value_sigma_to_bin_ratio: float = 0.75
    critic_mtp_horizon: int = 6

    normalize_reward: bool = False
    clip_reward: bool = False

    # --- HOPSD v4 ---
    hindsight_horizon: int = 20      # H: future window for the plan features (~1/(1-gamma*lambda))
    distill_coef: float = 1.0        # student actor loss = distill_coef * clipped forward KL
    distill_kl_clip: float = 2.0     # tau: pointwise per-action-dim KL clip (paper's min(l, tau))
    teacher_conc_cap: float = 100.0  # hard cap on decoder Beta concentrations (sane sharp targets)
    teacher_vf_coef: float = 0.5     # plan-critic CE weight inside the teacher update
    teacher_grad_clip: float = 0.5   # teacher's own global clip (separate optimizer)
    plan_eta: float = 0.25           # ascent step size, in units of the batch per-dim plan spread
    plan_steps: int = 4              # ascent steps per state per iteration
    plan_radius: float = 1.0         # trust region: ||(phi*-phi)/sigma_batch|| cap (sigma units)
    plan_move_std: bool = False      # ascend the plan std dims too (default off: their dV is
                                     # confounded — noise hurts return — and shrinking them
                                     # arms an entropy ratchet through the decoder)
    plan_chunk: int = 4096           # batch chunking for the ascent passes

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
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
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

    def get_value(self, x):
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.num_bins)

    def get_action_and_value(self, x, z=None):
        # z is the native Beta sample in (0,1); replaying it recomputes log_prob at the
        # same sample (the base's z-replay). Also returns the Beta params for distillation.
        actor_feat, critic_feat = self._trunks(x)
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
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
    """Decoder pi_T(a | s, phi) + plan critic V_T(s, phi); both take [obs, phi, valid_frac]."""

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        H = args.hidden
        plan_in = obs_dim + 2 * act_dim + 1
        self.actor_trunk = ThinkTrunk(plan_in, H, args.k_blocks, args.n_experts)
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.critic_trunk = ThinkTrunk(plan_in, H, args.k_blocks, args.n_experts)
        self.critic_head = layer_init(nn.Linear(H, args.num_bins, bias=False), std=0.1)
        with torch.no_grad():
            self.critic_head.weight.zero_()
        self.conc_cap = args.teacher_conc_cap

    def actor_params_for(self, x_plan):
        feat = self.actor_trunk(x_plan)
        alpha = (1.0 + F.softplus(self.alpha_head(feat))).clamp(max=self.conc_cap)
        beta = (1.0 + F.softplus(self.beta_head(feat))).clamp(max=self.conc_cap)
        return alpha, beta

    def critic_logits(self, x_plan):
        return self.critic_head(self.critic_trunk(x_plan))


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
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, _, _ = agent.get_action_and_value(next_obs)
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_transition_value_logits = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape)
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

            # --- Achieved plan features ---
            fut_mean, fut_std, fut_valid_frac = build_future_features(
                actions, transition_boundaries, args.hindsight_horizon
            )

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.critic_mtp_horizon, args.num_bins)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()
        b_phi0 = torch.cat([fut_mean, fut_std], dim=-1).reshape(-1, 2 * act_dim)
        b_vfrac = fut_valid_frac.reshape(-1, 1)
        b_plan_in0 = torch.cat([b_obs, b_phi0, b_vfrac], dim=-1)  # decoder/critic input @ achieved plan

        # --- Plan improvement: phi* = trust-region gradient ascent on V_T(s, phi) ---
        # Uses the pre-update plan critic (zero-init head at iteration 1 -> zero gradient
        # -> phi* = phi -> pure hindsight-BC cold start). Gradients flow only into phi;
        # torch.autograd.grad does not accumulate into teacher parameters.
        A = act_dim
        b_phi_star = b_phi0.clone()
        # Sigma-scaled trust region: ascent lengths are measured in units of the batch's
        # per-dim plan spread, so the counterfactual shrinks as the policy concentrates
        # (an absolute radius becomes many conditional sigmas off-manifold late in
        # training). Std dims are frozen unless plan_move_std.
        phi_scale = b_phi0.std(dim=0).clamp_min(1e-3)
        move_dims = torch.zeros_like(phi_scale)
        move_dims[:A] = 1.0
        if args.plan_move_std:
            move_dims[A:] = 1.0
        plan_delta_sum, plan_dv_sum, plan_n = 0.0, 0.0, 0
        for start in range(0, args.batch_size, args.plan_chunk):
            sl = slice(start, start + args.plan_chunk)
            o, p0, vfc = b_obs[sl], b_phi0[sl], b_vfrac[sl]
            move_mask = (vfc > 0).float() * move_dims        # (chunk, 2A)
            with torch.no_grad():
                v0 = hl_support.to_scalar(teacher.critic_logits(torch.cat([o, p0, vfc], dim=-1)))
            p = p0.clone()
            for _ in range(args.plan_steps):
                p_req = p.detach().requires_grad_(True)
                v = hl_support.to_scalar(teacher.critic_logits(torch.cat([o, p_req, vfc], dim=-1)))
                (grad_p,) = torch.autograd.grad(v.sum(), p_req)
                with torch.no_grad():
                    g_u = grad_p * phi_scale * move_mask     # gradient in sigma coords
                    gn = g_u.norm(dim=-1, keepdim=True)
                    step_u = args.plan_eta * g_u / (gn + 1e-8) * (gn > 0).float()
                    p = p + step_u * phi_scale               # back to raw coords
                    p[:, :A] = p[:, :A].clamp(-1.0, 1.0)     # plan means: valid action box
                    p[:, A:] = p[:, A:].clamp(0.0, 1.0)      # plan stds: non-negative, bounded
                    u = (p - p0) / phi_scale
                    un = u.norm(dim=-1, keepdim=True)
                    p = p0 + u * (args.plan_radius / (un + 1e-8)).clamp(max=1.0) * phi_scale
            with torch.no_grad():
                v1 = hl_support.to_scalar(teacher.critic_logits(torch.cat([o, p, vfc], dim=-1)))
                worse = (v1 < v0).unsqueeze(-1).float()      # ascent failed: revert per-sample
                p = worse * p0 + (1.0 - worse) * p
                b_phi_star[sl] = p
                plan_delta_sum += ((p - p0) / phi_scale).norm(dim=-1).sum().item()  # sigma units
                plan_dv_sum += (v1 - v0).clamp_min(0.0).sum().item()
                plan_n += p.shape[0]
        b_plan_in_star = torch.cat([b_obs, b_phi_star, b_vfrac], dim=-1)

        b_inds = np.arange(args.batch_size)
        distill_kls, distill_clipfracs, teacher_nlls = [], [], []
        # The target_kl leash freezes only the STUDENT; the decoder + plan critic always
        # train their full epochs (supervised losses, off-policy safe).
        student_stopped = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # ---- teacher update: decoder NLL @ achieved plan + plan-critic CE ----
                t_alpha0, t_beta0 = teacher.actor_params_for(b_plan_in0[mb_inds])
                t_nll = -Beta(t_alpha0, t_beta0).log_prob(b_latent_zs[mb_inds]).sum(-1)
                teacher_actor_loss = t_nll.mean()   # plain reconstruction, no weighting
                t_value_logits = teacher.critic_logits(b_plan_in0[mb_inds])
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

                # ---- student update: distill toward the decoder @ IMPROVED plan ----
                with torch.no_grad():
                    t_alpha, t_beta = teacher.actor_params_for(b_plan_in_star[mb_inds])
                _, _, newlogprob, entropy, value_logits, s_alpha, s_beta = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                kl_dims = beta_kl_per_dim(t_alpha, t_beta, s_alpha, s_beta)
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

        # Full-batch diagnostics (chunked, no grad): plan-critic EV, decoder entropy at
        # achieved vs improved plans, student entropy, mean-action gap (student vs
        # decoder @ phi*) in native z space.
        with torch.no_grad():
            t_vals, t_ents, t_ents_star, s_ents, gaps = [], [], [], [], []
            for start in range(0, args.batch_size, args.minibatch_size):
                sl = slice(start, start + args.minibatch_size)
                ta0, tb0 = teacher.actor_params_for(b_plan_in0[sl])
                t_ents.append(Beta(ta0, tb0).entropy().sum(-1).mean().item())
                ta, tb = teacher.actor_params_for(b_plan_in_star[sl])
                t_ents_star.append(Beta(ta, tb).entropy().sum(-1).mean().item())
                _, _, _, s_ent, _, sa, sb = agent.get_action_and_value(b_obs[sl], b_latent_zs[sl])
                s_ents.append(s_ent.mean().item())
                gaps.append((ta / (ta + tb) - sa / (sa + sb)).abs().mean().item())
                t_vals.append(hl_support.to_scalar(teacher.critic_logits(b_plan_in0[sl])))
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
        writer.add_scalar("debug/teacher_entropy_star", np.mean(t_ents_star), global_step)
        writer.add_scalar("debug/student_entropy", np.mean(s_ents), global_step)
        writer.add_scalar("debug/teacher_student_mean_gap", np.mean(gaps), global_step)
        writer.add_scalar("debug/plan_delta", plan_delta_sum / max(plan_n, 1), global_step)
        writer.add_scalar("debug/plan_dv", plan_dv_sum / max(plan_n, 1), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/fut_valid_frac", fut_valid_frac.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
