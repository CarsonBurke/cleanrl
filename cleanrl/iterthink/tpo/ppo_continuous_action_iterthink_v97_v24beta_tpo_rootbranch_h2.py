# PPO + IterThink v97 (v24 Beta + root-action TPO with H2 branch scores). From v96.
#
# This variant replaces v24's clipped PPO actor surrogate with Target Policy
# Optimization over a finite set of native action candidates sampled from the
# rollout policy at each state. Unlike v94's myopic immediate-reward score, v97
# treats each candidate as the first action of a short old-policy branch. It
# restores the simulator state, executes the candidate, then samples H-1 actions
# from the frozen rollout policy and scores the branch by dense discounted rewards
# plus a frozen critic bootstrap. TPO then constructs
# q_i ∝ p_old(a_i|s) exp(zscore(score_i) / eta) on the ROOT action simplex and
# fits the current root action probabilities by cross-entropy. PPO ratio/clip are
# kept only as diagnostics; they do not enter the actor loss.
#
# Hypothesis: dense rewards should enter TPO at every env step, but the compared
# candidate object must be temporally meaningful. Short branch returns preserve
# dense reward intervals while avoiding both v94's contextual-bandit failure and
# v95's tail-likelihood credit assignment error. v97 uses H=2 by default because
# H=4 was too slow for an 8M HalfCheetah run.
#
# --- inherited v24 notes ---
# PPO + IterThink v24 (ACTION-DISTRIBUTION TOGGLE: dreamer4-faithful). From v21.
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The distributional critic target is deliberately entropy-free so the fixed support
# remains calibrated. Off (default) => byte-identical to the v24 base.
#
# WHY v24. The v22/v23 state-dependent Gaussian std hit a 1/sigma^2 pathology
# (confident low-sigma states spike the mean gradient). dreamer4 avoids this two
# ways, and v24 ports BOTH faithfully behind one `--actor-dist` toggle, on the
# UNCHANGED v21 winner machinery (shared backbone, 2-way decoupled clip,
# rankgauss, clip-higher, tkl03) so the ONLY thing that varies is the action
# distribution — a clean A/B.
#
#   actor_dist="beta"  (DEFAULT, the "performs much better" path):
#       unimodal Beta, exactly dreamer4's continuous_dist_type='beta' (which
#       forces unimodal=True) and our beta_relusq:
#           alpha = 1 + softplus(head_a);  beta = 1 + softplus(head_b)   (>=1 => unimodal)
#       native support (0,1) is linearly rescaled to the env action range
#       [low, high]. Sampling clamps z to [eps, 1-eps]; log_prob/entropy are the
#       closed-form Beta values in native z-space (the constant rescale Jacobian
#       is dropped — it cancels in the PPO ratio and the entropy is a constant
#       offset). Bounded support => no squash saturation, no 1/sigma^2 blow-up,
#       no boundary mass leak, no bang-bang (unimodal).
#
#   actor_dist="gaussian"  (the matched control = state-dependent Gaussian scale):
#       dreamer4's Gaussian readout. This is NOT SAC's exact log-std head. It is a
#       state-dependent log-VARIANCE head (not a flat Parameter, not log-std),
#       SOFT-bounded by dreamer4's tanh-rescale (not a hard clamp, so the gradient
#       never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink/SAC tanh-squash + stable Jacobian on the sample (mean
#       stays raw). SAC continuous-action instead uses a state-dependent log_std
#       head bounded to [-5, 2] and std = exp(log_std). Here logvar [-8, 8] implies
#       log_std [-4, 4], so the family matches but the scale parameterization and
#       bounds do not.
#
# PARITY NOTES (both dists): the rollout buffers the distribution-NATIVE sample
# (latent_zs) — pre-tanh z for gaussian, z in (0,1) for beta — and replays it on
# the update pass, so log_prob is recomputed at the same sample (identical to
# v21's z-replay). `actions` holds the env action (tanh(z) / rescaled z). The
# gaussian path is bit-identical to v21 except the flat logstd -> dreamer4 head.
# Bar to beat: v21 flat-Gaussian = 8774.
#
# --- inherited v21 notes ---
# PPO + IterThink v21 (SHARED BACKBONE + DECOUPLED GRAD CLIP). From v19.
#
# WHY v21. v19 used two independent ThinkTrunks (one actor, one critic). The
# classic MuJoCo-PPO result is that shared backbones LOSE, because the value
# loss gradient dominates the shared trunk and corrupts the policy's features.
# v21 tests whether we can have the representation-sharing benefit WITHOUT that
# cost, by decoupling the gradient magnitudes:
#   - share_backbone: one ThinkTrunk feeds both the actor head and the
#     (distributional) critic head; trunk is computed once per forward.
#   - separate_grad_clip: DUAL-BACKWARD clipping. The value gradient
#     (vf_coef * v_loss) and the policy gradient (policy_loss - ent) are each
#     backpropped and clipped to their OWN max-norm (critic_grad_clip /
#     actor_grad_clip), then summed on the shared trunk:
#         trunk.grad = clip_actor(d pg / d trunk) + clip_critic(d vl / d trunk)
#     so the distributional critic's large CE gradient can no longer swamp the
#     shared features. NOTE: the trunk's effective budget is the SUM of the two
#     clips, so each defaults to 0.25 (sum ~= v19's single 0.5 global clip).
# This is targeted: rankgauss already bounds the POLICY gradient (rank-only adv),
# so the dominant imbalance on a shared trunk is the critic -> clip it apart.
# Built on the v19 winner: adv_transform="rankgauss" + clip-higher (0.2/0.28).
# Both knobs are toggles, so this file also runs the {shared,separate} x
# {global,decoupled-clip} 2x2. The bar to beat: rankgauss_cliphigh ~= 8292 (towers).
#
# --- inherited v19 notes ---
# PPO + IterThink v19 (ADVANTAGE SHAPING — magnitude-preserving + attribution). From v17.
#
# WHY v19. A subagent review of v17 (CDF-rank distributional PG) found that in its
# STABLE regime the categorical critic is overconfident, so u=F_Z(G) is bimodal at
# 0/1, the probit saturates, and the advantage DEGENERATES to ≈sign(GAE) (corr 0.92);
# norm_adv then re-standardizes the ±3.3 spikes to ≈±1 binary. So v17 discards the
# advantage MAGNITUDE (the thing PPO needs) and is really a sign-of-TD-error update
# made trainable by KL control. v17's 5867@4M conflates THREE possible causes — the
# distribution, a bounded/outlier-robust advantage, and KL control — introduced at
# once. v19 disentangles them and adds the principled fix, via one `adv_transform`:
#
#   "v10"      : raw GAE (== v10 / dist_pg off). Baseline.
#   "cdf_probit": v17's CDF-rank u -> Phi^-1(u). Reference.
#   "tanh_std" : A~ = tanh( GAE_t / (kappa * sigma(s_t)) ).  THE FIX. Per-state
#                normalized by the critic's return std sigma(s) (v16's good idea),
#                but BOUNDED by tanh (fixes v16's blowup: tiny sigma -> saturate, not
#                explode) AND magnitude-preserving near 0 (fixes v17's sign-collapse:
#                linear in GAE for |GAE|<kappa*sigma). Note G_t-E[Z_t]=GAE_t exactly.
#   "tanh_gae" : A~ = tanh( zscore(GAE)_t / kappa ).  Robust-GAE CONTROL with NO
#                distribution — isolates "bounded/outlier-robust advantage" from the
#                distributional claim. If this matches v17, the distribution is
#                incidental and this is the cleaner lever.
#
# All paths keep the mean-value GAE and the distributional λ-return value target
# (v10) UNCHANGED; only the policy advantage is reshaped. sigma(s) is the std of the
# OLD rollout Z(s_t), floored at `sigma_floor_bins` bins. Pair with target_kl for the
# 2x2 attribution (v10/tanh_gae/cdf_probit x KL-cap). Control: v17 / v10.
import os
import random
import time
from dataclasses import dataclass
from math import log
from typing import Optional

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

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def tpo_skill(scores, score_floor=0.0, eps=1e-6):
    centered = scores - scores.mean(dim=-1, keepdim=True)
    var = centered.var(dim=-1, keepdim=True, unbiased=False)
    denom = torch.sqrt(var + score_floor**2).clamp_min(eps)
    return centered / denom


def tpo_target(old_log_scores, scores, eta, score_floor=0.0):
    skill = tpo_skill(scores, score_floor=score_floor)
    return torch.softmax(F.log_softmax(old_log_scores, dim=-1) + skill / eta, dim=-1)


def tpo_cross_entropy_loss(new_log_scores, old_log_scores, scores, eta, score_floor=0.0):
    q = tpo_target(old_log_scores.detach(), scores.detach(), eta, score_floor=score_floor).detach()
    log_p = F.log_softmax(new_log_scores, dim=-1)
    return -(q * log_p).sum(dim=-1), q


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
    norm_adv: bool = True            # retained for v24 CLI compatibility; TPO standardizes per candidate group
    clip_coef: float = 0.2           # PPO-style diagnostic clip threshold only
    clip_coef_high: float = 0.28     # retained for v24 CLI compatibility; unused by pure TPO
    ent_coef: float = 0.0
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # candidate-simplex KL early-stop leash

    # v97: pure TPO actor objective over root actions, scored by short branches.
    # Candidate 0 is still the behavior action executed on the real wrapped env.
    tpo_group_size: int = 4
    tpo_branch_horizon: int = 2
    tpo_eta: float = 2.0
    tpo_score_floor: float = 0.1
    tpo_weighted_contexts: bool = False
    tpo_context_weight_min: float = 0.25
    tpo_context_weight_max: float = 4.0

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"  # retained for v24 CLI compatibility; unused by pure TPO

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # retained for v24 CLI compatibility; unused by pure TPO
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

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
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    cur = env
    while True:
        if isinstance(cur, wrapper_type):
            return cur
        if not hasattr(cur, "env"):
            return None
        cur = cur.env


def rms_after_one(mean, var, count, sample):
    batch_mean = sample
    batch_var = np.zeros_like(sample, dtype=np.float64)
    batch_count = 1
    delta = batch_mean - mean
    total_count = count + batch_count
    new_mean = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / total_count
    new_var = m2 / total_count
    return new_mean, new_var, total_count


def normalize_raw_obs(raw_obs, mean, var, eps):
    return np.clip((raw_obs - mean) / np.sqrt(var + eps), -10.0, 10.0)


def branch_tpo_rollouts(envs, agent, root_obs, first_zs, first_actions, first_logprobs, support, args, device):
    """Short old-policy branches from each real state.

    Returns:
      branch_obs: (N, K, H, obs_dim), normalized obs at each branch action.
      branch_zs: (N, K, H, act_dim), native action samples.
      branch_masks: (N, K, H), 1 while the branch is alive.
      branch_old_logprobs: (N, K), old policy log-probs for root actions.
      branch_scores: (N, K), discounted frozen-normalized reward sum plus bootstrap.
    """
    num_envs, num_candidates = first_logprobs.shape
    horizon = args.tpo_branch_horizon
    obs_dim = root_obs.shape[-1]
    act_dim = first_actions.shape[-1]
    branch_obs = torch.zeros((num_envs, num_candidates, horizon, obs_dim), device=device)
    branch_zs = torch.zeros((num_envs, num_candidates, horizon, act_dim), device=device)
    branch_masks = torch.zeros((num_envs, num_candidates, horizon), device=device)
    branch_old_logprobs = torch.zeros((num_envs, num_candidates), device=device)
    branch_scores = torch.zeros((num_envs, num_candidates), device=device)

    first_actions_np = first_actions.detach().cpu().numpy()

    states = {}
    root_states = {}
    obs_stats = {}
    reward_vars = {}
    reward_eps = {}
    done = np.zeros((num_envs, num_candidates), dtype=bool)
    raw_obs_next = [[None for _ in range(num_candidates)] for _ in range(num_envs)]

    gamma_powers = torch.tensor([args.gamma ** h for h in range(horizon)], dtype=torch.float32, device=device)

    for env_i, wrapped_env in enumerate(envs.envs):
        norm_obs_wrapper = find_wrapper(wrapped_env, gym.wrappers.NormalizeObservation)
        norm_rew_wrapper = find_wrapper(wrapped_env, gym.wrappers.NormalizeReward)
        time_limit_wrapper = find_wrapper(wrapped_env, gym.wrappers.TimeLimit)
        assert norm_obs_wrapper is not None and norm_rew_wrapper is not None
        base = wrapped_env.unwrapped
        root_qpos = base.data.qpos.copy()
        root_qvel = base.data.qvel.copy()
        root_time = float(base.data.time)
        root_states[env_i] = (root_qpos, root_qvel, root_time)
        obs_stats[env_i] = (
            norm_obs_wrapper.obs_rms.mean.copy(),
            norm_obs_wrapper.obs_rms.var.copy(),
            norm_obs_wrapper.epsilon,
        )
        reward_vars[env_i] = float(norm_rew_wrapper.return_rms.var)
        reward_eps[env_i] = norm_rew_wrapper.epsilon
        elapsed_steps = getattr(time_limit_wrapper, "_elapsed_steps", None)
        max_episode_steps = getattr(time_limit_wrapper, "_max_episode_steps", None)
        for cand_i in range(num_candidates):
            branch_obs[env_i, cand_i, 0] = root_obs[env_i]
            branch_zs[env_i, cand_i, 0] = first_zs[env_i, cand_i]
            branch_masks[env_i, cand_i, 0] = 1.0
            branch_old_logprobs[env_i, cand_i] = first_logprobs[env_i, cand_i]

            base.set_state(root_qpos, root_qvel)
            base.data.time = root_time
            raw_next_obs, raw_reward, terminated, truncated, _ = base.step(first_actions_np[env_i, cand_i])
            if elapsed_steps is not None and max_episode_steps is not None:
                truncated = truncated or (elapsed_steps + 1 >= max_episode_steps)

            norm_reward = np.clip(raw_reward / np.sqrt(reward_vars[env_i] + reward_eps[env_i]), -10.0, 10.0)
            branch_scores[env_i, cand_i] += gamma_powers[0] * float(norm_reward)
            done[env_i, cand_i] = bool(terminated or truncated)
            raw_obs_next[env_i][cand_i] = raw_next_obs
            states[(env_i, cand_i)] = (base.data.qpos.copy(), base.data.qvel.copy(), float(base.data.time))
        base.set_state(root_qpos, root_qvel)
        base.data.time = root_time

    for h in range(1, horizon):
        live_keys = [(env_i, cand_i) for env_i in range(num_envs) for cand_i in range(num_candidates) if not done[env_i, cand_i]]
        if not live_keys:
            break
        obs_batch_np = []
        for env_i, cand_i in live_keys:
            obs_mean, obs_var, obs_eps = obs_stats[env_i]
            obs_batch_np.append(normalize_raw_obs(raw_obs_next[env_i][cand_i], obs_mean, obs_var, obs_eps))
        obs_batch = torch.tensor(np.asarray(obs_batch_np), dtype=torch.float32, device=device)
        with torch.no_grad():
            action_batch, z_batch, _, _, _ = agent.get_action_and_value(obs_batch)

        action_batch_np = action_batch.detach().cpu().numpy()
        for j, (env_i, cand_i) in enumerate(live_keys):
            branch_obs[env_i, cand_i, h] = obs_batch[j]
            branch_zs[env_i, cand_i, h] = z_batch[j]
            branch_masks[env_i, cand_i, h] = 1.0

            wrapped_env = envs.envs[env_i]
            time_limit_wrapper = find_wrapper(wrapped_env, gym.wrappers.TimeLimit)
            elapsed_steps = getattr(time_limit_wrapper, "_elapsed_steps", None)
            max_episode_steps = getattr(time_limit_wrapper, "_max_episode_steps", None)
            base = wrapped_env.unwrapped
            qpos, qvel, sim_time = states[(env_i, cand_i)]
            base.set_state(qpos, qvel)
            base.data.time = sim_time
            raw_next_obs, raw_reward, terminated, truncated, _ = base.step(action_batch_np[j])
            if elapsed_steps is not None and max_episode_steps is not None:
                truncated = truncated or (elapsed_steps + h + 1 >= max_episode_steps)

            norm_reward = np.clip(raw_reward / np.sqrt(reward_vars[env_i] + reward_eps[env_i]), -10.0, 10.0)
            branch_scores[env_i, cand_i] += gamma_powers[h] * float(norm_reward)
            done[env_i, cand_i] = bool(terminated or truncated)
            raw_obs_next[env_i][cand_i] = raw_next_obs
            states[(env_i, cand_i)] = (base.data.qpos.copy(), base.data.qvel.copy(), float(base.data.time))

    bootstrap_keys = [(env_i, cand_i) for env_i in range(num_envs) for cand_i in range(num_candidates) if not done[env_i, cand_i]]
    if bootstrap_keys:
        obs_batch_np = []
        for env_i, cand_i in bootstrap_keys:
            obs_mean, obs_var, obs_eps = obs_stats[env_i]
            obs_batch_np.append(normalize_raw_obs(raw_obs_next[env_i][cand_i], obs_mean, obs_var, obs_eps))
        obs_batch = torch.tensor(np.asarray(obs_batch_np), dtype=torch.float32, device=device)
        with torch.no_grad():
            value_probs = torch.softmax(agent.get_value(obs_batch), dim=-1)
            boot_values = (value_probs * support).sum(dim=-1)
        for j, (env_i, cand_i) in enumerate(bootstrap_keys):
            branch_scores[env_i, cand_i] += (args.gamma ** horizon) * boot_values[j]

    # Restore root states after all branch probes.
    for env_i, wrapped_env in enumerate(envs.envs):
        base = wrapped_env.unwrapped
        root_qpos, root_qvel, root_time = root_states[env_i]
        base.set_state(root_qpos, root_qvel)
        base.data.time = root_time

    return branch_obs, branch_zs, branch_masks, branch_old_logprobs, branch_scores


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
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution is sharp at 0 (not uniform),
        # preventing the distributional-bootstrap blowup that sank v9.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            z = torch.linspace(args.v_min, args.v_max, args.num_bins)
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
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
        # Returns (dist, to_action, log_det_fn) where:
        #   to_action(z): map a NATIVE sample z to the env action.
        #   log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
        #                  from dist.log_prob(z) (0 where the map is volume-constant).
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
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
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

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
        value_logits = self.critic_head(critic_feat)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, value_logits

    def get_action_candidates_and_value(self, x, num_candidates):
        # Sample a finite TPO candidate set from the rollout policy. Candidate 0
        # is the behavior action that enters the env; all candidates are stored
        # in native distribution space and replayed during optimization.
        actor_feat, critic_feat = self._trunks(x)
        candidate_feat = actor_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        dist, to_action, log_det_fn = self._actor_dist(candidate_feat)
        z_candidates = dist.sample()
        if self.actor_dist == "beta":
            z_candidates = z_candidates.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        candidate_log_probs = (dist.log_prob(z_candidates) - log_det_fn(z_candidates)).sum(-1)
        z = z_candidates[:, 0]
        candidate_actions = to_action(z_candidates)
        action = candidate_actions[:, 0]
        value_logits = self.critic_head(critic_feat)
        if self.actor_dist == "gaussian":
            dist_1, _, log_det_1 = self._actor_dist(actor_feat)
            zr = dist_1.rsample()
            entropy = (dist_1.log_prob(zr) - log_det_1(zr)).sum(1).neg()
        else:
            entropy = dist.entropy()[:, 0].sum(1)
        return action, z, z_candidates, candidate_actions, candidate_log_probs, entropy, value_logits

    def get_candidate_log_probs_and_value(self, x, z_candidates):
        # Re-score the stored root-action candidate set under the current policy.
        # The TPO simplex is over first actions at the real rollout state; short
        # branches only provide candidate scores.
        actor_feat, critic_feat = self._trunks(x)
        num_candidates = z_candidates.shape[1]
        candidate_feat = actor_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        dist, _, log_det_fn = self._actor_dist(candidate_feat)
        candidate_log_probs = (dist.log_prob(z_candidates) - log_det_fn(z_candidates)).sum(-1)
        value_logits = self.critic_head(critic_feat)
        if self.actor_dist == "gaussian":
            dist_1, _, log_det_1 = self._actor_dist(actor_feat)
            zr = dist_1.rsample()
            entropy = (dist_1.log_prob(zr) - log_det_1(zr)).sum(1).neg()
        else:
            entropy = dist.entropy()[:, 0].sum(1)
        return candidate_log_probs, entropy, value_logits

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are pre-clamped to [v_min, v_max]).
    """
    n = support.shape[0]
    tz = atoms.clamp(v_min, v_max)
    b = (tz - v_min) / bin_width                       # (B, n) fractional bin pos
    lo = b.floor()
    hi = b.ceil()
    lo_idx = lo.clamp(0, n - 1).long()
    hi_idx = hi.clamp(0, n - 1).long()
    m = torch.zeros_like(probs)
    m.scatter_add_(1, lo_idx, probs * (hi - b))        # mass to lower bin
    m.scatter_add_(1, hi_idx, probs * (b - lo))        # mass to upper bin
    # When b is integer, lo == hi and both weights are 0; (1 - (hi - lo)) == 1
    # routes the full mass to that bin. Otherwise hi - lo == 1 → adds 0.
    m.scatter_add_(1, lo_idx, probs * (1.0 - (hi - lo)))
    return m


def distributional_lambda_returns(
    rewards, dones, next_done, value_probs, bootstrap_probs, support, v_min, v_max, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return G^λ (probs per step).

        G^λ_t =_D r_t + γ·nonterm·[ (1-λ)·Z(s_{t+1}) + λ·G^λ_{t+1} ]

    Mean-matches the scalar GAE λ-return. Shapes: rewards/dones (T, B);
    value_probs (T, B, n); bootstrap_probs (B, n) = Z(s_T). Returns (T, B, n).
    Entropy/soft-value terms are NOT injected here — the critic regresses to the raw
    reward return; max-ent enters the policy advantage separately (see --auto-entropy).
    """
    T = rewards.shape[0]
    target = torch.zeros_like(value_probs)
    g_next = bootstrap_probs                            # G^λ_{T} ≡ bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done               # (B,)
            z_next = bootstrap_probs                    # Z(s_T)
        else:
            nonterminal = 1.0 - dones[t + 1]
            z_next = value_probs[t + 1]                 # Z(s_{t+1})
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n)
        gn = (gamma * nonterminal).unsqueeze(-1)        # (B, 1)
        atoms = rewards[t].unsqueeze(-1) + gn * support  # (B, n) transformed atoms
        g_next = categorical_project(mix, atoms, support, v_min, v_max, bin_width)
        target[t] = g_next
    return target


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform. Works on a full
    batch or a single minibatch (sigma/u must be sliced to match gae)."""
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (tanh_gae kappa=1 > kappa=2). Smaller kappa => harder.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        return torch.tanh(z / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        # Sign-correct WITHOUT count distortion: take plain rankgauss's GLOBAL-rank
        # magnitude, then force the sign to match the raw advantage. Fixes the flaw in
        # rankgauss_signed (per-group half-Gaussian over-amplifies the minority sign by
        # COUNT); here magnitude still reflects global rank extremity and only the ~9%
        # near-zero "flips" get re-signed. Nonlinear (not a shift) => survives norm_adv.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    assert args.tpo_group_size >= 2, "TPO needs at least two scored candidates"
    assert args.tpo_branch_horizon >= 1, "TPO branch horizon must be positive"
    assert args.tpo_eta > 0.0, "tpo_eta must be positive"
    assert args.tpo_score_floor >= 0.0, "TPO score floor must be non-negative"
    assert args.env_id == "HalfCheetah-v4", "v97 counterfactual probing is HalfCheetah-v4-specific"
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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        0.5,  # sigma_ratio unused (categorical Bellman target, no Gaussian projection)
        device,
        use_symlog=args.value_symlog,
    )
    support = hl_support.support                       # (num_bins,) linear support
    bin_width = hl_support.bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    branch_obs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_group_size, args.tpo_branch_horizon)
        + envs.single_observation_space.shape
    ).to(device)
    branch_zs = torch.zeros(
        (args.num_steps, args.num_envs, args.tpo_group_size, args.tpo_branch_horizon)
        + envs.single_action_space.shape
    ).to(device)
    branch_masks = torch.zeros((args.num_steps, args.num_envs, args.tpo_group_size, args.tpo_branch_horizon)).to(device)
    branch_logprobs = torch.zeros((args.num_steps, args.num_envs, args.tpo_group_size)).to(device)
    branch_scores = torch.zeros((args.num_steps, args.num_envs, args.tpo_group_size)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, _, z_cands, cand_actions, cand_logprobs, ent, value_logits = agent.get_action_candidates_and_value(
                    next_obs, args.tpo_group_size
                )
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = (p * support).sum(dim=-1)
                br_obs, br_zs, br_masks, br_logprobs, br_scores = branch_tpo_rollouts(
                    envs, agent, next_obs, z_cands, cand_actions, cand_logprobs, support, args, device
                )
            branch_obs[step] = br_obs
            branch_zs[step] = br_zs
            branch_masks[step] = br_masks
            branch_logprobs[step] = br_logprobs
            branch_scores[step] = br_scores
            logprobs[step] = cand_logprobs[:, 0]

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

        with torch.no_grad():
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, boot_logits = agent.get_action_and_value(next_obs)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                bootstrap_probs = torch.softmax(agent.get_value(next_obs), dim=-1)   # (B, n) = Z(s_T)
                next_value_bonus = None
            next_value = (bootstrap_probs * support).sum(dim=-1).reshape(1, -1)
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
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
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * (nextvalues + next_value_bonus[t]) * nextnonterminal - values[t]
                    policy_adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            else:
                policy_adv = advantages
            # Critic target: RAW reward λ-return (entropy-free => no support overflow).
            target_probs = distributional_lambda_returns(
                rewards, dones, next_done, value_probs, bootstrap_probs,
                support, args.v_min, args.v_max, bin_width, args.gamma, args.gae_lambda,
            )
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = ((returns.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_branch_obs = branch_obs.reshape(
            (-1, args.tpo_group_size, args.tpo_branch_horizon) + envs.single_observation_space.shape
        )
        b_branch_zs = branch_zs.reshape(
            (-1, args.tpo_group_size, args.tpo_branch_horizon) + envs.single_action_space.shape
        )
        b_branch_masks = branch_masks.reshape(-1, args.tpo_group_size, args.tpo_branch_horizon)
        b_branch_logprobs = branch_logprobs.reshape(-1, args.tpo_group_size)
        b_branch_scores = branch_scores.reshape(-1, args.tpo_group_size)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        # Diagnostics: keep the inherited v24 advantage transform visible. The
        # actor target itself uses short branch returns collected during rollout.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        mean_abs_adv = b_policy_adv.abs().mean().clamp_min(1e-6)
        if args.tpo_weighted_contexts:
            b_context_weights = (b_policy_adv.abs() / mean_abs_adv).clamp(
                args.tpo_context_weight_min, args.tpo_context_weight_max
            ).detach()
        else:
            b_context_weights = torch.ones_like(b_policy_adv)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()
        branch0 = b_branch_scores[:, 0]
        bz = (branch0 - branch0.mean()) / (branch0.std() + 1e-8)
        branch_gae_corr = (bz * az).mean().item()
        candidate0_rank = (b_branch_scores <= branch0.unsqueeze(-1)).float().mean(dim=-1).mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        group_kls = []
        old_executed_kls = []
        executed_kls = []
        tpo_q_entropies = []
        tpo_q_maxes = []
        tpo_score_stds = []
        tpo_context_weight_means = []
        tpo_context_weight_maxes = []
        stop_update = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_branch_zs = b_branch_zs[mb_inds]
                new_root_logprobs, entropy_loss, value_logits = agent.get_candidate_log_probs_and_value(
                    b_obs[mb_inds], mb_branch_zs[:, :, 0]
                )
                entropy_loss = entropy_loss.mean()
                newlogprob = new_root_logprobs[:, 0]
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_executed_kl = (-logratio).mean()
                    executed_approx_kl = ((ratio - 1) - logratio).mean()
                    old_group_logp = F.log_softmax(b_branch_logprobs[mb_inds], dim=-1)
                    new_group_logp = F.log_softmax(new_root_logprobs, dim=-1)
                    approx_kl = (old_group_logp.exp() * (old_group_logp - new_group_logp)).sum(dim=-1).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    group_kls.append(approx_kl.item())
                    old_executed_kls.append(old_executed_kl.item())
                    executed_kls.append(executed_approx_kl.item())

                tpo_scores = b_branch_scores[mb_inds]
                per_context_tpo_loss, tpo_q = tpo_cross_entropy_loss(
                    new_root_logprobs,
                    b_branch_logprobs[mb_inds],
                    tpo_scores,
                    args.tpo_eta,
                    score_floor=args.tpo_score_floor,
                )
                context_weights = b_context_weights[mb_inds]
                policy_loss = (context_weights * per_context_tpo_loss).mean()
                with torch.no_grad():
                    tpo_q_entropies.append((-(tpo_q * tpo_q.clamp_min(1e-12).log()).sum(dim=-1)).mean().item())
                    tpo_q_maxes.append(tpo_q.max(dim=-1).values.mean().item())
                    tpo_score_stds.append(tpo_scores.std(dim=-1, unbiased=False).mean().item())
                    tpo_context_weight_means.append(context_weights.mean().item())
                    tpo_context_weight_maxes.append(context_weights.max().item())
                # Distributional value loss: cross-entropy to the (fixed)
                # distributional λ-return target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic's CE gradient
                    # cannot swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (policy_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = policy_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/tpo_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        writer.add_scalar("losses/old_approx_kl", np.mean(old_executed_kls), global_step)
        writer.add_scalar("losses/approx_kl", np.mean(group_kls), global_step)
        writer.add_scalar("losses/executed_approx_kl", np.mean(executed_kls), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("debug/tpo_target_entropy", np.mean(tpo_q_entropies), global_step)
        writer.add_scalar("debug/tpo_target_max_prob", np.mean(tpo_q_maxes), global_step)
        writer.add_scalar("debug/tpo_group_score_std", np.mean(tpo_score_stds), global_step)
        writer.add_scalar("debug/tpo_score_floor", args.tpo_score_floor, global_step)
        writer.add_scalar("debug/tpo_context_weight_mean", np.mean(tpo_context_weight_means), global_step)
        writer.add_scalar("debug/tpo_context_weight_max", np.mean(tpo_context_weight_maxes), global_step)
        writer.add_scalar("debug/tpo_candidate0_rank", candidate0_rank, global_step)
        writer.add_scalar("debug/tpo_branch_return_vs_gae_corr", branch_gae_corr, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
