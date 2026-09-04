# TD7-HOP v2 — widen the privileged-teacher channel that v1 validated.
# =====================================================================================
# v1 RESULT (HalfCheetah, seed 1, eval/checkpoint returns at matched steps vs td7_v1):
#   explore arm +8% at 130k (12,034 vs 11,138) — hindsight privilege pays through DATA;
#   sdpg -12% (replacing the deterministic operator with a Beta student is a tax);
#   pure -62% (distillation as the SOLE operator is rate-limited: distill_kl sat BELOW the
#   1.2-nat budget with lam floored while teacher_q_gap stayed > 0 — the teacher led, the
#   student followed, but the two-stage improve->distill chain is simply slower than DPG).
# v2 keeps the winning explore substrate (TD7 actor/critic byte-intact, teacher+student as
# the eps-mixed behavior policy) and adds two INDEPENDENT ways to extract more from the
# privileged teacher, one per arm:
#   --guide-coef 1.0 (v2_guide): advantage-gated teacher->actor distillation. Each actor
#     update also draws a uniform batch (which carries phi), samples guide_k actions from
#     the privileged teacher, ranks them with the min-twin critic, and pulls the actor by
#     MSE toward the frontier-softmax-weighted mix of candidates that BEAT the actor's own
#     Q — improving-only and ramped, so the term is inert wherever the teacher has nothing
#     the critic endorses. Privilege now reaches the OPERATOR through a critic-gated
#     channel (what sdpg wanted) without giving up the deterministic actor (why sdpg lost).
#   --explore-eps 0.4 --hop-update-freq 1 (v2_press): more dose through the exact channel
#     that worked — 40% of behavior steps from the student instead of 25%, teacher/student
#     trained every step instead of every other, so the student tracks a fresher teacher.
# HYPOTHESIS: v1_explore's gain is hindsight-directed state-correlated behavior feeding the
# critic/LAP. If so, dose scales it (press) and feeding the same critic-endorsed proposals
# to the actor compounds it (guide). v1_explore keeps running as the control.
# ------------------------------------------------------------------------------------
# (v1 header follows; substrate and teacher/student machinery otherwise unchanged.)
# The hopsd line's architectural core — a teacher pi_T(a|s, phi) conditioned on the realized
# FUTURE (phi = pooled near/far future-action window), trained to rationalize good outcomes
# (advantage-weighted NLL) and to propose beyond them (Q-ascent + sampled search), distilled
# into a phi-free student under a dual-controlled followability budget — was capped by its
# on-policy PPO skeleton (best 6.8k@1M vs TD7's 16k@1M). This file transplants that core onto
# TD7's substrate (SALE frozen-embedding twin critic, LAP replay, target clipping,
# checkpointing — all byte-preserved) to test whether the teacher-student factoring ADDS to a
# strong off-policy engine or was only compensating for PPO's weaknesses.
#   TEACHER pi_T(a|s, phi): Beta policy (conc cap 300), TD7-actor-style trunk over [s, phi]
#     + fixed_zs. phi = near/far mean-std pooling of the next H=20 executed actions +
#     valid_frac (25 dims) — the proven-followable, replay-storable privilege (hopsd v19/v36);
#     built by a delay queue at insert time, windows never cross episode boundaries, partials
#     flush at the boundary BEFORE the training burst. Trained every policy_freq steps on a
#     UNIFORM replay minibatch (LAP's priority bias would concentrate the tilt on exactly the
#     critic's worst states and detune the KL-targeted temperature):
#     (a) hindsight AWR: w * NLL(z_decision), adv = minQ(s, a_exec) - minQ(s, a_deployed(s)),
#         weights via hopsd v12.2's robust dual (winsorize adv_z at 2 sigma, bisect softmax
#         temp to tilt KL = 1.2 nats). NLL is at the stored PRE-NOISE decision latent z, not
#         the clamped executed action — clamping Beta samples + noise creates boundary atoms
#         whose NLL gradient (~|log eps|) would dominate the batch.
#     (b) proposal operators (ramped over search_ramp_steps; frontier-normalized by a
#         fixed-radius 0.1 Q-sensitivity EMA — the coverage-INDEPENDENT scale, hopsd v34.1):
#         v41 sampled search (k Beta samples ranked by min-twin Q with per-candidate zsa,
#         softmax in frontier units, weighted NLL toward its own Q-best proposals) + v40 DPG
#         through the Beta rsample. NO grad-ratio ceiling: coupling the proposal dose to the
#         self-shrinking AWR grad norm re-creates the v34 self-extinguishing normalizer; the
#         KL dual is the ONE governor (v40's thesis taken seriously).
#     (c) followability: lam * KL(pi_T || pi_S) (student detached), lam dual-updated every
#         target_update_rate steps (offset 125 from the hard swaps) toward teacher_kl_budget.
#   STUDENT pi_S(a|s): Beta policy, trained ONLY by clipped per-dim forward KL(pi_T || pi_S)
#     at the stored phi (hopsd's exact distillation loss). Deployment/eval/checkpoint/TD
#     targets use the Beta MEAN; behavior is an epsilon-mix (mean + TD7's 0.1 noise with prob
#     1-eps, raw Beta sample with prob eps) so state-correlated exploration enters WITHOUT an
#     uncontrolled entropy drag through the 25k-250k window where TD7 builds ~80% of its lead.
#   ARMS (flags; one mechanism per run):
#     pure (defaults)          — student = pure distillation. The v40 thesis on a real
#                                substrate; scientific control. Expected fastest falsifier:
#                                distill_kl pinned at budget + teacher_q_gap > 0 + slope <
#                                td7_v1 = "operator alive but rate-limited".
#     --student-dpg-coef 0.5   — flagship: student gets frontier-normalized DPG through its
#                                own rsample IN ADDITION to distillation. Floor near TD7's
#                                operator; the teacher becomes a hindsight prior/data shaper.
#     --explore-only           — TD7's deterministic actor, DPG loss, targets, checkpointing
#                                all byte-preserved as the improvement engine and eval policy;
#                                the teacher+distilled student exist ONLY as the eps-mixed
#                                behavior component. Privilege enters exclusively through
#                                which data gets collected — the channel where off-policy TD
#                                is agnostic and hindsight-informed state-correlated
#                                exploration is genuinely novel. Floor ~= td7_v1 by design.
#   NOT included, with reasons: hindsight-critic value distillation (pooled mean/std of the
#     future actions determine E||a||^2 exactly, and HalfCheetah reward = v_x - 0.1||a||^2,
#     so Q_T(s,a,phi_realized) reads off the realized return — regressing the twins toward it
#     is a Monte-Carlo-return regression in disguise, variance-neutral at best vs TD7's
#     clipped 1-step target); g/return features in any teacher input (v2 fixed point).
# HYPOTHESIS: on a substrate that already climbs, hindsight privilege pays through data
#   (explore arm) or as a prior (sdpg arm) more than as the sole operator (pure arm). Gates:
#   td7_v1 logged 12,755@250k / 16,056@1M — any arm below ~8k@250k is off-trajectory (kill);
#   watch charts-vs-eval return gap + student entropy (entropy drag), boundary_z_frac
#   (atom fitting), distill_kl vs budget + teacher_q_gap (rate-limit signature).
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
import tyro
from torch.utils.tensorboard import SummaryWriter

SAMPLE_EPS = 1e-6  # clamp Beta latents off the open-interval boundary (avoid log(0))
NEAR_HORIZON = 5   # near window = a_{t+1..t+5}; far window = a_{t+6..t+H} (hopsd v19)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # TD7 substrate (unchanged defaults)
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments (TD7 requires 1)"""
    learning_starts: int = 25000
    """timesteps of uniform-random actions before training starts"""
    use_checkpoints: bool = True
    """train in episode-boundary bursts and evaluate the checkpointed best-worst-case policy"""
    eval_freq: int = 5000
    """evaluate every N env steps"""
    eval_eps: int = 10
    """number of evaluation episodes"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update_rate: int = 250
    """hard-update the target networks and fixed encoders every N training steps"""
    exploration_noise: float = 0.1
    """the scale of exploration noise (on the deployed-mean behavior branch)"""
    target_policy_noise: float = 0.2
    """the scale of target policy smoothing noise"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    policy_freq: int = 2
    """the frequency of the TD7 actor update (delayed policy update); teacher + student run
    on their own hop_update_freq cadence"""
    lap_alpha: float = 0.4
    """LAP prioritization exponent"""
    min_priority: float = 1.0
    """LAP minimum priority (and Huber loss threshold)"""
    max_eps_when_checkpointing: int = 20
    """episodes to assess a policy before checkpointing (after steps_before_checkpointing)"""
    steps_before_checkpointing: int = 750000
    """training steps of early exploration before full-length checkpoint assessment begins"""
    reset_weight: float = 0.9
    """discount applied to best_min_return when switching to full checkpoint assessment"""
    zs_dim: int = 256
    """dimensionality of the SALE embeddings"""
    hidden_dim: int = 256
    """hidden layer width of all networks"""
    encoder_lr: float = 3e-4
    """the learning rate of the encoder optimizer"""
    critic_lr: float = 3e-4
    """the learning rate of the critic optimizer"""
    actor_lr: float = 3e-4
    """the learning rate of the policy-side optimizers (student, teacher, [actor])"""

    # --- arms ---
    student_dpg_coef: float = 0.0
    """student additionally ascends min-twin Q through its own Beta rsample (frontier-
    normalized). 0 = pure-distillation student (the v40 thesis); 0.5 = the flagship arm."""
    explore_only: bool = False
    """keep TD7's deterministic actor/DPG/targets/checkpointing byte-intact as the improvement
    engine and eval policy; teacher+student act only through the eps-mixed behavior policy"""

    # --- privilege (phi) ---
    hindsight_horizon: int = 20
    """H: future-action window for phi (~ the GAE effective horizon the hopsd line used)"""

    # --- teacher ---
    teacher_conc_cap: float = 300.0
    """Beta concentration cap (bang-bang headroom; also the sharpness ceiling)"""
    tilt_eps: float = 1.2
    """target KL(softmax tilt || uniform) of the hindsight AWR weights, in nats"""
    adv_clip: float = 2.0
    """winsorize the z-scored advantage here BEFORE the tilt (robust dual, hopsd v12.2)"""
    search_coef: float = 0.5
    """weight of the sampled-search NLL term (teacher proposal operator)"""
    search_k: int = 8
    """teacher Beta samples per state for the search operator"""
    search_tau: float = 1.0
    """softmax temperature for the search weights, in q_frontier units"""
    teacher_dpg_coef: float = 0.5
    """weight of the teacher DPG (Q-ascent through the Beta rsample) term"""
    search_ramp_steps: int = 200000
    """training steps to ramp the proposal operators (search + DPG) linearly from 0"""
    qadv_floor: float = 0.05
    """floor on the q_frontier normalizer (fixed-radius Q action-sensitivity EMA)"""
    teacher_kl_budget: float = 1.2
    """followability budget: target mean KL(pi_T || pi_S) in nats (the ONE governor of how
    far the teacher may lead; there is deliberately no grad-ratio ceiling)"""
    teacher_kl_eta: float = 0.05
    """dual-ascent step. hopsd used 0.5 ONCE per 32k-step iteration; this dual fires every
    250 training steps (~130x more often), so the per-update gain is scaled down to keep the
    aggregate controller gain comparable (still ~13x faster engagement than hopsd)"""
    teacher_grad_clip: float = 0.5
    """teacher global grad clip (hopsd treated this as load-bearing: boundary-adjacent NLL
    rows can spike the AWR term)"""
    teacher_kl_lam_init: float = 0.1
    teacher_kl_lam_min: float = 0.02
    teacher_kl_lam_max: float = 100.0

    # --- student / behavior ---
    distill_kl_clip: float = 2.0
    """tau: pointwise per-action-dim clip on the forward distillation KL"""
    explore_eps: float = 0.25
    """behavior mixes: deployed mean + exploration_noise with prob 1-eps, a raw student Beta
    sample with prob eps (state-correlated exploration without an uncontrolled entropy drag)"""
    hop_update_freq: int = 2
    """cadence (training steps) of the teacher+student updates. v1 tied this to policy_freq
    (=2); 1 doubles distillation throughput so the student tracks a fresher teacher"""

    # --- v2: teacher-guided actor distillation (explore-only arms) ---
    guide_coef: float = 0.0
    """weight of the advantage-gated teacher->actor MSE distillation (0 = off). Improving-
    only: only teacher proposals whose min-twin Q beats the actor's own action contribute"""
    guide_k: int = 8
    """teacher Beta samples per state for the guidance term"""
    guide_tau: float = 1.0
    """softmax temperature over improving candidates, in q_frontier units"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.pow(2), min_priority * td_loss).sum(1).mean()


def beta_kl_per_dim(a1, b1, a2, b2):
    """Forward KL( Beta(a1,b1) || Beta(a2,b2) ) per action dim, closed form (hopsd)."""

    def ln_beta_fn(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    return (
        ln_beta_fn(a2, b2)
        - ln_beta_fn(a1, b1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


def awr_weights(adv, tilt_eps, adv_clip):
    """hopsd v12.2 robust KL-targeted tilt: winsorize the z-scored advantage FIRST (outliers
    cannot eat the budget), then geometric-bisect the softmax temperature so the tilt sits at
    tilt_eps nats from uniform. Returns (mean-1 weights, temp). No grad."""
    adv_z = (adv - adv.mean()) / (adv.std() + 1e-8)
    a = adv_z.clamp(-adv_clip, adv_clip)
    n = float(a.numel())
    lo, hi = 0.02, 50.0
    for _ in range(25):
        mid = (lo * hi) ** 0.5
        p = torch.softmax(a / mid, dim=0)
        tilt_kl = (p * (p * n).clamp_min(1e-12).log()).sum().item()
        if tilt_kl > tilt_eps:
            lo = mid
        else:
            hi = mid
    temp = (lo * hi) ** 0.5
    w = torch.softmax(a / temp, dim=0) * n
    return w, temp


def dual_update_lam(lam, kl, budget, eta, lam_min, lam_max):
    """One multiplicative dual-ascent step for the followability multiplier (hopsd v40_1)."""
    lam = lam * math.exp(eta * (kl / budget - 1.0))
    return float(min(max(lam, lam_min), lam_max))


def pooled_phi(future_actions, horizon):
    """hopsd v19 pooled privilege from a window of future executed actions, shape (m, A) with
    m <= H (possibly m=0): [near_mean, near_std, far_mean, far_std, valid_frac] (4A+1).
    Population std; zeros for an empty window; valid_frac over the full horizon."""
    m, A = future_actions.shape
    parts = []
    for w in (future_actions[:NEAR_HORIZON], future_actions[NEAR_HORIZON:]):
        if len(w) > 0:
            parts.append(w.mean(0))
            parts.append(w.std(0))
        else:
            parts.append(np.zeros(A, dtype=np.float32))
            parts.append(np.zeros(A, dtype=np.float32))
    parts.append(np.array([m / float(horizon)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PhiQueue:
    """Delay queue building phi at insert time from a single env's stream.

    A transition's phi pools the NEXT `horizon` executed actions; windows never cross an
    episode boundary. A record is released once its full window exists; at a boundary all
    pending records flush with partial windows (before the training burst fires in the main
    loop, so no data is missing when training starts)."""

    def __init__(self, horizon, act_dim):
        self.h = horizon
        self.act_dim = act_dim
        self.pending = []  # [state, action, z, next_state, reward, done, n_future_seen]
        self.future = []   # future[i] is pending[i]'s own executed action (kept aligned)

    def push(self, state, action, z, next_state, reward, done, boundary):
        # `action` is the executed env action in [-1, 1]; it is a FUTURE action for every
        # already-pending record and the base action of the new record.
        for rec in self.pending:
            rec[6] += 1
        self.future.append(np.array(action, copy=True))
        self.pending.append(
            [np.array(state, copy=True), np.array(action, copy=True), np.array(z, copy=True),
             np.array(next_state, copy=True), float(reward), float(done), 0]
        )
        out = []
        if boundary:
            for i, (s, a, zz, ns, r, d, m) in enumerate(self.pending):
                fut = np.array(self.future[i + 1 :], dtype=np.float32).reshape(-1, self.act_dim)
                out.append((s, a, zz, ns, r, d, pooled_phi(fut, self.h)))
            self.pending.clear()
            self.future.clear()
        elif self.pending[0][6] >= self.h:
            s, a, zz, ns, r, d, m = self.pending.pop(0)
            fut = np.array(self.future[1 : self.h + 1], dtype=np.float32).reshape(-1, self.act_dim)
            out.append((s, a, zz, ns, r, d, pooled_phi(fut, self.h)))
            self.future.pop(0)
        return out


class LAPBuffer:
    """TD7's LAP buffer + two columns for the hindsight machinery: the pre-noise decision
    latent z (Beta NLL target — the executed action develops clamp atoms at the rails) and
    the pooled privilege phi. sample() (LAP, critic path) is unchanged; sample_uniform()
    serves the teacher/student updates (LAP's priority bias would concentrate the AWR tilt
    on the critic's worst-modeled states and detune the KL-targeted temperature)."""

    def __init__(self, state_dim, action_dim, phi_dim, device, max_size, batch_size):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.z = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.phi = np.zeros((self.max_size, phi_dim), dtype=np.float32)

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

    def add(self, state, action, z, next_state, reward, done, phi):
        # `action` is already normalized to [-1, 1] (the caller divides by max_action once,
        # since the same normalized action also feeds the phi pooling)
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.z[self.ptr] = z
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.phi[self.ptr] = phi

        self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        csum = torch.cumsum(self.priority[: self.size], 0)
        val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
        self.ind = torch.searchsorted(csum, val).cpu().data.numpy()

        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device),
        )

    def sample_uniform(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)
        return (
            torch.tensor(self.state[ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[ind], dtype=torch.float, device=self.device),
            torch.tensor(self.z[ind], dtype=torch.float, device=self.device),
            torch.tensor(self.phi[ind], dtype=torch.float, device=self.device),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class Actor(nn.Module):
    """TD7's deterministic actor (used only by --explore-only)."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class BetaPolicy(nn.Module):
    """Beta policy with TD7's actor trunk shape: AvgL1Norm(l0(inputs)) cat zs -> 2 hidden ->
    alpha/beta heads (1 + softplus, capped). Used for the student (inputs = s) and the
    teacher (inputs = [s, phi])."""

    def __init__(self, in_dim, action_dim, zs_dim=256, hdim=256, conc_cap=300.0):
        super().__init__()
        self.conc_cap = conc_cap
        self.l0 = nn.Linear(in_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.alpha_head = nn.Linear(hdim, action_dim)
        self.beta_head = nn.Linear(hdim, action_dim)

    def forward(self, inputs, zs):
        h = avg_l1_norm(self.l0(inputs))
        h = torch.cat([h, zs], 1)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        alpha = (1.0 + F.softplus(self.alpha_head(h))).clamp(max=self.conc_cap)
        beta = (1.0 + F.softplus(self.beta_head(h))).clamp(max=self.conc_cap)
        return alpha, beta

    def mean_action(self, inputs, zs):
        alpha, beta = self.forward(inputs, zs)
        return 2.0 * (alpha / (alpha + beta)) - 1.0


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        zs = avg_l1_norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = avg_l1_norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2 = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class TD7HopAgent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer
        self.action_dim = int(action_dim)
        phi_dim = 4 * self.action_dim + 1

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.critic = Critic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        # student: the phi-free Beta policy (in every arm; in explore-only it is behavior-only)
        self.student = BetaPolicy(state_dim, action_dim, args.zs_dim, args.hidden_dim,
                                  args.teacher_conc_cap).to(device)
        self.student_optimizer = torch.optim.Adam(self.student.parameters(), lr=args.actor_lr)

        # teacher: privileged Beta policy over [s, phi]
        self.teacher = BetaPolicy(state_dim + phi_dim, action_dim, args.zs_dim, args.hidden_dim,
                                  args.teacher_conc_cap).to(device)
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=args.actor_lr)

        # deployed-policy networks per arm
        if args.explore_only:
            self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
            self.actor_target = copy.deepcopy(self.actor)
            self.checkpoint_policy = copy.deepcopy(self.actor)
        else:
            self.actor = None
            self.student_target = copy.deepcopy(self.student)
            self.checkpoint_policy = copy.deepcopy(self.student)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, phi_dim, device,
                                       args.buffer_size, args.batch_size)

        self.max_action = max_action
        self.training_steps = 0

        # checkpointing tracked values (TD7 machinery, unchanged)
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # target value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0.0
        self.min_target = 0.0

        # hindsight machinery state
        self.q_frontier = None            # EMA of fixed-radius 0.1 Q action-sensitivity (raw units)
        self.lam_kl = args.teacher_kl_lam_init
        self.kl_accum = 0.0               # mean-KL accumulator between dual updates
        self.kl_count = 0
        self.hop_stats = {}               # last hop-update telemetry
        self.guide_stats = {}             # last guidance-term telemetry

    # ---------- action selection ----------
    def _deployed_action(self, state, zs):
        """The deterministic deployed/eval policy at `state` (live networks)."""
        if self.args.explore_only:
            return self.actor(state, zs)
        return self.student.mean_action(state, zs)

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        """Returns (env action, decision latent z in (0,1)). The decision latent is the
        pre-noise policy decision the teacher will rationalize (Beta sample on the sample
        branch; the deployed mean otherwise)."""
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                if self.args.explore_only:
                    action = self.checkpoint_policy(state, zs)
                else:
                    action = self.checkpoint_policy.mean_action(state, zs)
                decision = action
            else:
                zs = self.fixed_encoder.zs(state)
                if use_exploration and np.random.rand() < self.args.explore_eps:
                    # state-correlated exploration: raw student Beta sample (no extra noise)
                    alpha, beta = self.student(state, zs)
                    z = torch.distributions.Beta(alpha, beta).sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    decision = 2.0 * z - 1.0
                    action = decision
                else:
                    decision = self._deployed_action(state, zs)
                    action = decision
                    if use_exploration:
                        action = action + torch.randn_like(action) * self.args.exploration_noise

            z_decision = ((decision.clamp(-1, 1) + 1.0) / 2.0).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            return (
                action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action,
                z_decision.cpu().data.numpy().flatten(),
            )

    # ---------- helpers ----------
    def _min_q(self, state, action, zs):
        zsa = self.fixed_encoder.zsa(zs, action)
        return self.critic(state, action, zsa, zs).min(1)[0]

    # ---------- training ----------
    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # update encoder (TD7, unchanged)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # update critic (TD7, unchanged except the target policy is the deployed one)
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            if self.args.explore_only:
                next_base = self.actor_target(next_state, fixed_target_zs)
            else:
                next_base = self.student_target.mean_action(next_state, fixed_target_zs)
            next_action = (next_base + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(
                1, keepdim=True
            )[0]
            Q_target = reward + not_done * self.args.gamma * Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update LAP priorities (unchanged)
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # ---------- policy side (delayed): teacher -> student at hop cadence; the TD7
        # actor keeps its own byte-preserved policy_freq cadence ----------
        if self.training_steps % self.args.hop_update_freq == 0:
            self._hop_update()
        if self.args.explore_only and self.training_steps % self.args.policy_freq == 0:
            self._actor_update(state, fixed_zs)

        # hard target/fixed-encoder updates + target clip range snapshot (TD7, unchanged)
        if self.training_steps % self.args.target_update_rate == 0:
            if self.args.explore_only:
                self.actor_target.load_state_dict(self.actor.state_dict())
            else:
                self.student_target.load_state_dict(self.student.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        # followability dual update — offset 125 from the hard swaps so the dual never FIRES
        # on the exact swap step (its accumulation window still straddles one swap; the
        # averaging over ~125 policy updates absorbs the transient)
        if self.training_steps % self.args.target_update_rate == self.args.target_update_rate // 2:
            if self.kl_count > 0:
                self.lam_kl = dual_update_lam(
                    self.lam_kl, self.kl_accum / self.kl_count, self.args.teacher_kl_budget,
                    self.args.teacher_kl_eta, self.args.teacher_kl_lam_min, self.args.teacher_kl_lam_max,
                )
            self.kl_accum = 0.0
            self.kl_count = 0

        if self.training_steps % 500 == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", Q.mean().item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            self.writer.add_scalar("debug/lam_kl", self.lam_kl, self.training_steps)
            for tag, val in {**self.hop_stats, **self.guide_stats}.items():
                self.writer.add_scalar(f"debug/{tag}", val, self.training_steps)

    def _hop_update(self):
        args = self.args
        # UNIFORM minibatch for the hindsight machinery (never the LAP-prioritized one).
        u_state, u_action, u_z, u_phi = self.replay_buffer.sample_uniform()

        with torch.no_grad():
            zs_u = self.fixed_encoder.zs(u_state)

            # hindsight AWR advantage: executed action vs the current deployed policy, both
            # under the fixed embeddings. Detached — a weighting, never a gradient path.
            q_exec = self._min_q(u_state, u_action, zs_u)
            a_dep = self._deployed_action(u_state, zs_u)
            q_dep = self._min_q(u_state, a_dep, zs_u)
            w, auto_temp = awr_weights(q_exec - q_dep, args.tilt_eps, args.adv_clip)

            # fixed-radius (0.1) Q action-sensitivity EMA: the coverage-independent frontier
            # scale for the proposal operators (hopsd v34.1's lesson).
            u_dir = torch.randn_like(u_action)
            u_dir = u_dir / u_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            a_probe = (u_action + 0.1 * u_dir).clamp(-1, 1)
            sens = (self._min_q(u_state, a_probe, zs_u) - q_exec).abs().mean().item()
            self.q_frontier = sens if self.q_frontier is None else 0.99 * self.q_frontier + 0.01 * sens
            frontier_norm = max(self.q_frontier, args.qadv_floor)

            # student conditional (detached anchor for the followability penalty)
            s_alpha_d, s_beta_d = self.student(u_state, zs_u)

        # ---- teacher update ----
        t_in = torch.cat([u_state, u_phi], dim=1)
        t_alpha, t_beta = self.teacher(t_in, zs_u)
        t_dist = torch.distributions.Beta(t_alpha, t_beta)
        nll = -t_dist.log_prob(u_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)).sum(-1)
        awr_loss = (w * nll).mean()

        ramp = min(1.0, self.training_steps / max(1, args.search_ramp_steps))

        # v41 sampled search: k detached samples from the CURRENT teacher, ranked by min-twin
        # Q (per-candidate zsa), softmax in frontier units, weighted NLL toward the Q-best.
        with torch.no_grad():
            z_k = t_dist.sample((args.search_k,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)  # (k, B, A)
            a_k = 2.0 * z_k - 1.0
            B = u_state.shape[0]
            state_rep = u_state.unsqueeze(0).expand(args.search_k, B, -1).reshape(args.search_k * B, -1)
            zs_rep = zs_u.unsqueeze(0).expand(args.search_k, B, -1).reshape(args.search_k * B, -1)
            q_k = self._min_q(state_rep, a_k.reshape(args.search_k * B, -1), zs_rep).view(args.search_k, B)
            w_k = torch.softmax((q_k - q_k.max(0, keepdim=True).values) / (args.search_tau * frontier_norm), dim=0)
            search_ess = (1.0 / (w_k * w_k).sum(0).clamp_min(1e-8)).mean().item()
            search_top_gap = (q_k.max(0).values - q_dep).mean().item()
        logp_k = torch.distributions.Beta(t_alpha, t_beta).log_prob(z_k).sum(-1)  # (k, B), grad -> teacher
        search_loss = -(w_k * logp_k).sum(0).mean()

        # v40 teacher DPG: ascend min-twin Q through the Beta rsample, frontier-normalized.
        t_z_rs = t_dist.rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        t_a_rs = 2.0 * t_z_rs - 1.0
        q_t_rs = self._min_q(u_state, t_a_rs, zs_u)
        teacher_dpg = -q_t_rs.mean() / frontier_norm

        # followability: the ONE governor of teacher lead (no grad-ratio ceiling).
        teacher_kl = beta_kl_per_dim(t_alpha, t_beta, s_alpha_d, s_beta_d).clamp_min(0.0).sum(-1).mean()

        teacher_loss = (
            awr_loss
            + ramp * (args.search_coef * search_loss + args.teacher_dpg_coef * teacher_dpg)
            + self.lam_kl * teacher_kl
        )
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        nn.utils.clip_grad_norm_(self.teacher.parameters(), args.teacher_grad_clip)
        self.teacher_optimizer.step()

        self.kl_accum += float(teacher_kl.detach())
        self.kl_count += 1

        # ---- student update: clipped per-dim forward KL toward the (detached) teacher ----
        s_alpha, s_beta = self.student(u_state, zs_u)
        kl_dims = beta_kl_per_dim(t_alpha.detach(), t_beta.detach(), s_alpha, s_beta).clamp_min(0.0)
        distill_loss = kl_dims.clamp(max=args.distill_kl_clip).sum(-1).mean()
        student_loss = distill_loss
        if args.student_dpg_coef > 0.0:
            s_z_rs = torch.distributions.Beta(s_alpha, s_beta).rsample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            s_a_rs = 2.0 * s_z_rs - 1.0
            student_loss = student_loss + args.student_dpg_coef * (
                -self._min_q(u_state, s_a_rs, zs_u).mean() / frontier_norm
            )
        self.student_optimizer.zero_grad()
        student_loss.backward()
        self.student_optimizer.step()

        with torch.no_grad():
            self.hop_stats = {
                "distill_kl": float(kl_dims.sum(-1).mean()),
                "distill_clipfrac": float((kl_dims > args.distill_kl_clip).float().mean()),
                "teacher_kl": float(teacher_kl),
                "teacher_q_gap": float(q_t_rs.mean() - q_exec.mean()),
                "search_top_gap": search_top_gap,
                "search_ess": search_ess,
                "q_frontier": float(self.q_frontier),
                "auto_temp": float(auto_temp),
                "awr_ess": float(1.0 / ((w / w.sum()).pow(2).sum() * w.numel())),
                "teacher_entropy": float(t_dist.entropy().sum(-1).mean()),
                "student_entropy": float(torch.distributions.Beta(s_alpha, s_beta).entropy().sum(-1).mean()),
                "boundary_z_frac": float(
                    ((u_z < 1e-3) | (u_z > 1.0 - 1e-3)).float().mean()
                ),
                "action_boundary_frac": float((u_action.abs() > 0.95).float().mean()),
            }

    def _actor_update(self, lap_state, lap_fixed_zs):
        """TD7's actor DPG (byte-preserved, LAP minibatch) + optional advantage-gated
        teacher->actor distillation on a fresh uniform minibatch (which carries phi)."""
        args = self.args
        actor_action = self.actor(lap_state, lap_fixed_zs)
        actor_fixed_zsa = self.fixed_encoder.zsa(lap_fixed_zs, actor_action)
        actor_Q = self.critic(lap_state, actor_action, actor_fixed_zsa, lap_fixed_zs)
        actor_loss = -actor_Q.mean()

        if args.guide_coef > 0.0:
            g_state, _, _, g_phi = self.replay_buffer.sample_uniform()
            with torch.no_grad():
                zs_g = self.fixed_encoder.zs(g_state)
            a_pred = self.actor(g_state, zs_g)  # grad-carrying; detached copy feeds the gate
            with torch.no_grad():
                t_alpha, t_beta = self.teacher(torch.cat([g_state, g_phi], dim=1), zs_g)
                z_g = torch.distributions.Beta(t_alpha, t_beta).sample((args.guide_k,)).clamp(
                    SAMPLE_EPS, 1.0 - SAMPLE_EPS
                )
                a_g = 2.0 * z_g - 1.0  # (k, B, A)
                B = g_state.shape[0]
                s_rep = g_state.unsqueeze(0).expand(args.guide_k, B, -1).reshape(args.guide_k * B, -1)
                zs_rep = zs_g.unsqueeze(0).expand(args.guide_k, B, -1).reshape(args.guide_k * B, -1)
                q_g = self._min_q(s_rep, a_g.reshape(args.guide_k * B, -1), zs_rep).view(args.guide_k, B)
                q_act = self._min_q(g_state, a_pred.detach(), zs_g)
                frontier = max(self.q_frontier if self.q_frontier is not None else args.qadv_floor,
                               args.qadv_floor)
                # improving-only gate: candidates that beat the actor's own Q; columns with
                # no improving candidate contribute nothing (and get uniform dummy weights
                # to keep the softmax NaN-free)
                imp = q_g > q_act.unsqueeze(0)  # (k, B)
                has_imp = imp.any(0)
                logits = (q_g - q_g.max(0, keepdim=True).values) / (args.guide_tau * frontier)
                logits = logits.masked_fill(~imp, float("-inf"))
                safe_logits = torch.where(has_imp.unsqueeze(0), logits, torch.zeros_like(logits))
                w_g = torch.softmax(safe_logits, dim=0)
                a_bar = (w_g.unsqueeze(-1) * a_g).sum(0)  # (B, A)
                imp_f = has_imp.float()
                # frontier-proportional scale (raw Q per unit action): keeps the guidance a
                # CONSTANT fraction of the DPG gradient as Q grows — an unscaled action-MSE
                # is ~5% of the DPG gradient at 130k and decays further (review finding)
                guide_scale = frontier / 0.1
                self.guide_stats = {
                    "guide_imp_frac": float(imp_f.mean()),
                    "guide_gap": float(((q_g.max(0).values - q_act) * imp_f).sum()
                                       / imp_f.sum().clamp_min(1.0)),
                    "guide_scale": float(guide_scale),
                }
            ramp = min(1.0, self.training_steps / max(1, args.search_ramp_steps))
            guide_mse = (((a_pred - a_bar) ** 2).sum(-1) * imp_f).mean()
            actor_loss = actor_loss + ramp * args.guide_coef * guide_scale * guide_mse
            self.guide_stats["guide_mse"] = float(guide_mse.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    # ---------- TD7 checkpointing machinery (unchanged) ----------
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        if self.min_return < self.best_min_return:
            self.train_and_reset()

        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            source = self.actor if self.args.explore_only else self.student
            self.checkpoint_policy.load_state_dict(source.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(agent: TD7HopAgent, eval_env, eval_eps, use_checkpoint):
    returns = np.zeros(eval_eps)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(
                np.array(state), use_checkpoint=use_checkpoint, use_exploration=False
            )
            state, reward, terminated, truncated, _ = eval_env.step(action)
            returns[ep] += reward
            done = terminated or truncated
    return returns.mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "TD7 requires num_envs=1 (1:1 train/env-step ratio and episodic checkpointing)"
    assert args.guide_coef == 0.0 or args.explore_only, "--guide-coef requires --explore-only (it guides the TD7 actor)"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    eval_env = gym.make(args.env_id)
    eval_env.action_space.seed(args.seed + 100)

    state_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_action = float(envs.single_action_space.high[0])

    agent = TD7HopAgent(state_dim, action_dim, max_action, args, device, writer)
    phi_queue = PhiQueue(args.hindsight_horizon, action_dim)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    eval_seeded = False
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in TD7)
    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            if not eval_seeded:
                eval_env.reset(seed=args.seed + 100)
                eval_seeded = True
            eval_return = evaluate(agent, eval_env, args.eval_eps, use_checkpoint=args.use_checkpoints)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        # ALGO LOGIC: put action logic here
        if allow_train:
            act_env, z_dec = agent.select_action(np.array(obs[0]))
            actions = act_env[None]
        else:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            z_dec = np.clip((actions[0] / max_action + 1.0) / 2.0, SAMPLE_EPS, 1.0 - SAMPLE_EPS)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # phi delay queue -> replay buffer; handle `final_observation`. TD7's done quirk kept
        # (bootstrap at the timeout step even on true termination there). The queue holds a
        # transition until its 20-step future window exists; ALL pending records flush at a
        # boundary, which precedes the training burst below.
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        boundary = bool(terminations[0] or truncations[0])
        done = float(terminations[0] and not truncations[0])
        exec_action_norm = actions[0] / max_action
        for s, a, zz, ns, r, d, phi in phi_queue.push(
            obs[0], exec_action_norm, z_dec, real_next_obs, float(rewards[0]), done, boundary
        ):
            agent.replay_buffer.add(s, a, zz, ns, r, d, phi)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training (per-step when not checkpointing)
        if allow_train and not args.use_checkpoints:
            agent.train()

        # episode boundary: log return, run burst training/checkpointing, enable training
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_return = float(info["episode"]["r"])
                    ep_length = int(info["episode"]["l"])
                    print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    if allow_train and args.use_checkpoints:
                        agent.maybe_train_and_checkpoint(ep_length, ep_return)

                    if global_step >= args.learning_starts:
                        allow_train = True
                    break

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        save_dict = {
            "encoder": agent.encoder.state_dict(),
            "critic": agent.critic.state_dict(),
            "student": agent.student.state_dict(),
            "teacher": agent.teacher.state_dict(),
            "checkpoint_policy": agent.checkpoint_policy.state_dict(),
            "checkpoint_encoder": agent.checkpoint_encoder.state_dict(),
        }
        if agent.actor is not None:
            save_dict["actor"] = agent.actor.state_dict()
        torch.save(save_dict, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()
