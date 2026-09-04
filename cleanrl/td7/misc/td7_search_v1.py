# TD7-SEARCH v1 — TD7 + the validated improvement-operator ideas from the hopsd line.
# =====================================================================================
# Base: td7_continuous_action.py (TD3 + SALE + LAP + checkpointing + target clipping), which
# empirically dominates the entire hopsd PPO-distillation lineage on HalfCheetah (~14k@315k vs
# the line's best 6.8k@1M). v1 transplants the three mechanisms from that line (and its review)
# with the best evidence/fit, each independently flagged for attribution; ALL DEFAULTS OFF —
# flags off is byte-equivalent to baseline TD7 (the n-step machinery reduces exactly to the
# original buffer semantics at n=1).
#   (1) TRAIN-TIME SAMPLED SEARCH (--train-search) — hopsd v41's confirmed keeper (its only
#       arm to beat its own ledger), adapted to a deterministic actor. On each delayed actor
#       update: sample k perturbations of the actor's action (radius search_sigma — the radius
#       target-policy smoothing already trains the critic at), rank candidates INCLUDING the
#       incumbent by min(Q1,Q2) under the fixed SALE embeddings, softmax-weight the samples
#       that BEAT the incumbent by advantage/(tau*q_frontier), and pull the actor toward their
#       weighted mean by MSE. A global(ish) improvement step that supplies direction quality
#       where the local DPG slope is noisy; EXACT no-op at states where no sample beats the
#       incumbent (the advantage gate — a deterministic-MSE translation of v41's "uniform
#       weights = MLE toward own samples = no-op"). The DPG term is divided by the same
#       q_frontier scale (fixed-radius Q action-sensitivity EMA, floored) so the DPG:search
#       gradient ratio stays stationary as Q grows ~10x over training (the v42 value-scale
#       insight applied to operator conditioning, not targets); a measured grad-norm-ratio
#       ceiling (v40's guardrail, refreshed every target_update_rate steps) defuses windup
#       when Q flattens and the frontier rides its floor; search_coef ramps over
#       search_ramp_steps so garbage early Q never drags TD7's early velocity.
#   (2) ACT-TIME BEST-OF-K (--act-search-k > 0) — QT-Opt/GRAC-style action selection on the
#       BEHAVIOR policy only: propose k perturbations of the actor output, act on the
#       min(Q1,Q2)-argmax (exploration noise added after). Eval/checkpoint policy stays the
#       pure actor, so eval semantics are unchanged; improvement must flow through better
#       data. Zero training-dynamics risk, and the cleanest falsification test for (1): if
#       acting on the same radius-0.2 proposals does not lift behavior returns, the sampled
#       "headroom" is critic error and train-time search is chasing noise.
#   (3) N-STEP TD TARGETS (--q-nstep 3) — from hopsd v40 (and BRO/SR-SAC): faster credit
#       propagation, composing with TD7's double pessimism (min-twin + target clip) by
#       shortening bootstrap reliance. The buffer's not_done column is generalized to a
#       bootstrap-discount column disc = gamma^m * (1 - terminated_at_close) built by a
#       per-env accumulator that respects TD7's done quirk (bootstrap through truncation at
#       the real final obs; windows never cross episode boundaries; partials flush at the
#       boundary BEFORE the training burst). LAP priority/Huber stay in raw units — the
#       deliberate v42 lesson (its value-scale normalizer silently cancelled LAP until
#       priorities were rescaled back to raw units; we keep raw scale everywhere instead).
#   NOT transplanted, with reasons: the pessimism dial (TD7's actor already ascends the twin
#   MEAN — critic:258, actor:-cat(q1,q2).mean() — so "mean" isn't a change where it matters,
#   and a mean BOOTSTRAP would let one transient spike permanently inflate the monotone
#   target-clip ratchet); value-scale target normalization (collides with TD7's raw-scale
#   LAP/Huber/clip machinery); the whole hindsight teacher/distillation apparatus (its line
#   lost to TD7 2x at 4x the steps).
# HYPOTHESIS: (1) lifts the actor's ability to climb the SALE-conditioned ridge at matched
#   steps (aliveness: debug/search_improve_frac and debug/search_top_gap stay material);
#   (2) lifts behavior returns iff the headroom is real; (3) lifts early-mid sample
#   efficiency. Gate: matched-step eval vs td7_v1's logged curve at 500k/1M/2M; kill on
#   sustained underperformance, no sunk-cost extensions.
import copy
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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments (TD7 requires 1, see td7_continuous_action.py)"""
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
    """the scale of exploration noise"""
    target_policy_noise: float = 0.2
    """the scale of target policy smoothing noise"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    policy_freq: int = 2
    """the frequency of training the policy (delayed)"""
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
    """the learning rate of the actor optimizer"""

    # --- v1 (1): train-time sampled-search operator (hopsd v41 transplant) ---
    train_search: bool = False
    """add the sampled-search term to the delayed actor update"""
    search_k: int = 8
    """perturbation candidates per state (the incumbent actor action is always candidate 0)"""
    search_sigma: float = 0.2
    """candidate perturbation std (= target_policy_noise: the radius the critic is trained at)"""
    search_tau: float = 1.0
    """softmax temperature for the candidate advantages, in q_frontier units"""
    search_coef: float = 0.5
    """weight of the search MSE term after the ramp"""
    search_ramp_steps: int = 200000
    """training steps to ramp search_coef linearly from 0 (early Q is garbage; don't drag)"""
    search_ceiling: float = 2.0
    """ceiling on the measured ||g_search||/||g_dpg|| grad-norm ratio (v40 guardrail; fires
    only above the ceiling, refreshed every target_update_rate training steps)"""
    qadv_floor: float = 0.05
    """floor on the q_frontier normalizer (fixed-radius Q action-sensitivity EMA); load-bearing:
    when Q flattens the search tilt fades to uniform instead of amplifying noise"""

    # --- v1 (2): act-time best-of-k Q-argmax on the behavior policy ---
    act_search_k: int = 0
    """behavior-policy candidates per env step (0 disables). Selection by min(Q1,Q2) under the
    fixed embeddings; exploration noise is added AFTER selection. Eval/checkpoint unaffected."""

    # --- v1 (3): n-step TD targets ---
    q_nstep: int = 1
    """n-step horizon for the replay TD target (1 = baseline TD7, byte-equivalent semantics)"""


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


class NStepAccumulator:
    """Builds n-step records from a single env's transition stream.

    Each pending window accumulates gamma-discounted rewards; a window closes when it spans n
    transitions or the episode ends (termination OR truncation — windows never cross a reset),
    whichever comes first. The closing transition supplies the bootstrap state (the REAL next
    obs — final_observation at boundaries) and the bootstrap discount
        disc = gamma^m * (1 - terminated_at_close)
    where terminated_at_close applies TD7's done quirk (terminations and not truncations: the
    original bootstraps at the timeout step even when the env truly terminated there). At n=1
    every non-boundary step closes immediately with (R=r, disc=gamma), matching the original
    not_done column exactly. All pending windows flush at a boundary, so the buffer is complete
    before the episode-boundary training burst fires.

    Records also carry the true 1-STEP next state (the window's first transition's real next
    obs): the SALE encoder must keep its original 1-step latent-dynamics target even when the
    critic bootstraps at s_{t+m} — retargeting the encoder to a mixed-horizon prediction would
    silently degrade the representation everything else conditions on.
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.pending = []  # [state, action, next_state_1step, R, m], R = sum_{i<m} gamma^i r_{t+i}

    def push(self, state, action, reward, real_next_state, terminated_q, boundary):
        self.pending.append(
            [np.array(state, copy=True), np.array(action, copy=True), np.array(real_next_state, copy=True), 0.0, 0]
        )
        for rec in self.pending:
            rec[3] += (self.gamma ** rec[4]) * reward
            rec[4] += 1
        out = []
        if boundary:
            for s, a, n1, R, m in self.pending:
                out.append((s, a, n1, real_next_state, R, (self.gamma ** m) * (1.0 - terminated_q)))
            self.pending.clear()
        elif self.pending[0][4] >= self.n:
            s, a, n1, R, m = self.pending.pop(0)
            out.append((s, a, n1, real_next_state, R, self.gamma ** m))
        return out


class LAPBuffer:
    """LAP replay buffer: priorities live on-device, sampling by inverse-CDF over the priority
    cumsum. v1 change vs the TD7 base: the not_done column is generalized to a bootstrap
    discount `disc` (= gamma * not_done at n=1), so the TD target is reward + disc * Q'."""

    def __init__(self, state_dim, action_dim, device, max_size, batch_size, max_action):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state1 = np.zeros((self.max_size, state_dim), dtype=np.float32)  # 1-step (encoder target)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)   # n-step bootstrap state
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.disc = np.zeros((self.max_size, 1), dtype=np.float32)

        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = 1.0

        self.normalize_actions = float(max_action)

    def add(self, state, action, next_state1, next_state, reward, disc):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state1[self.ptr] = next_state1
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.disc[self.ptr] = disc

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
            torch.tensor(self.next_state1[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.disc[self.ind], dtype=torch.float, device=self.device),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())


class Actor(nn.Module):
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


class TD7Agent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer

        self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = LAPBuffer(state_dim, action_dim, device, args.buffer_size, args.batch_size, max_action)

        self.max_action = max_action
        self.training_steps = 0

        # checkpointing tracked values
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

        # search-operator state
        self.q_frontier = None       # EMA of mean |Q(s, a+sigma*eps) - Q(s, a_pi)| (raw Q units)
        self.search_atten = 1.0      # cached grad-ratio guardrail attenuation
        self.search_stats = {}       # last actor-update search telemetry
        self.act_gap_ema = 0.0       # EMA of the act-time argmax gap (behavior best-of-k)
        self.huber_linear_frac = 0.0

    def _min_q(self, state, action, zs):
        """min over twins of the critic under the fixed (non-target) embeddings."""
        zsa = self.fixed_encoder.zsa(zs, action)
        return self.critic(state, action, zsa, zs).min(1)[0]

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)
                # act-time best-of-k Q-argmax on the BEHAVIOR policy only — the use_exploration
                # gate keeps every eval rollout (checkpointed or not) the pure actor. Candidates
                # = incumbent + k perturbations at search_sigma; selection by min-twin Q under
                # the fixed embeddings; exploration noise added after.
                if self.args.act_search_k > 0 and use_exploration:
                    k = self.args.act_search_k
                    cands = (action + torch.randn(k, action.shape[1], device=self.device)
                             * self.args.search_sigma).clamp(-1, 1)
                    all_a = torch.cat([action, cands], 0)                      # (k+1, A)
                    state_rep = state.expand(k + 1, -1)
                    zs_rep = zs.expand(k + 1, -1)
                    q = self._min_q(state_rep, all_a, zs_rep)                  # (k+1,)
                    best = int(q.argmax())
                    self.act_gap_ema = 0.999 * self.act_gap_ema + 0.001 * float(q[best] - q[0])
                    action = all_a[best : best + 1]

            if use_exploration:
                action = action + torch.randn_like(action) * self.args.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def _search_term(self, state, fixed_zs, actor_action):
        """Sampled-search operator: advantage-gated weighted MSE toward the candidates that
        beat the incumbent by min-twin Q. Returns (search_mse, frontier_norm); records
        telemetry in self.search_stats. Candidates/weights carry no grad."""
        B, A = actor_action.shape
        k = self.args.search_k
        with torch.no_grad():
            a0 = actor_action.detach()
            cands = (a0.unsqueeze(0) + torch.randn(k, B, A, device=self.device)
                     * self.args.search_sigma).clamp(-1, 1)                    # (k, B, A)
            all_a = torch.cat([a0.unsqueeze(0), cands], 0)                     # (k+1, B, A)
            state_rep = state.unsqueeze(0).expand(k + 1, B, -1).reshape((k + 1) * B, -1)
            zs_rep = fixed_zs.unsqueeze(0).expand(k + 1, B, -1).reshape((k + 1) * B, -1)
            q_all = self._min_q(state_rep, all_a.reshape((k + 1) * B, A), zs_rep).view(k + 1, B)
            adv = q_all[1:] - q_all[:1]                                        # (k, B)

            # fixed-radius Q action-sensitivity EMA (raw units) — the shared scale for the
            # search softmax AND the DPG term, so operator strength is Q-scale-invariant.
            sens = adv.abs().mean().item()
            self.q_frontier = sens if self.q_frontier is None else 0.99 * self.q_frontier + 0.01 * sens
            frontier_norm = max(self.q_frontier, self.args.qadv_floor)

            improving = adv > 0
            has_imp = improving.any(0)                                         # (B,)
            logits = torch.where(improving, adv / (self.args.search_tau * frontier_norm),
                                 torch.full_like(adv, float("-inf")))
            # states with no improving candidate: give safe logits, mask weights to 0 after
            safe_logits = torch.where(has_imp.unsqueeze(0), logits, torch.zeros_like(adv))
            w = torch.softmax(safe_logits, 0) * has_imp.unsqueeze(0)           # (k, B)
            a_bar = (w.unsqueeze(-1) * cands).sum(0)                           # (B, A)

            imp_f = has_imp.float()
            n_imp = imp_f.sum().clamp(min=1.0)
            ess = ((1.0 / w.pow(2).sum(0).clamp_min(1e-8)) * imp_f).sum() / n_imp
            self.search_stats = {
                "search_top_gap": float((q_all.max(0)[0] - q_all[0]).mean()),
                "search_improve_frac": float(imp_f.mean()),
                "search_ess": float(ess),
                "q_frontier": float(self.q_frontier),
            }
        search_mse = (((actor_action - a_bar) ** 2).sum(-1) * imp_f).mean()
        return search_mse, frontier_norm

    def train(self):
        self.training_steps += 1

        state, action, next_state1, next_state, reward, disc = self.replay_buffer.sample()

        # update encoder: predict the 1-STEP next state's embedding from (zs, action) — always
        # the original SALE objective, even when the critic bootstraps at the n-step state.
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state1)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # update critic. disc = gamma^m * (1 - terminated_at_close) carries the whole
        # bootstrap coefficient (n-step aware; = gamma * not_done at n=1).
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(
                1, keepdim=True
            )[0]
            Q_target = reward + disc * Q_target.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = lap_huber(td_loss, self.args.min_priority)
        if self.training_steps % 500 == 0:  # host sync only at the logging cadence
            self.huber_linear_frac = float((td_loss >= self.args.min_priority).float().mean())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update LAP priorities
        priority = td_loss.max(1)[0].clamp(min=self.args.min_priority).pow(self.args.lap_alpha).detach()
        self.replay_buffer.update_priority(priority)

        # update actor (delayed)
        if self.training_steps % self.args.policy_freq == 0:
            actor_action = self.actor(state, fixed_zs)
            actor_fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
            actor_Q = self.critic(state, actor_action, actor_fixed_zsa, fixed_zs)

            if not self.args.train_search:
                actor_loss = -actor_Q.mean()
            else:
                search_mse, frontier_norm = self._search_term(state, fixed_zs, actor_action)
                # DPG in frontier units: keeps the DPG:search gradient ratio stationary as the
                # raw Q scale grows (Adam is ~invariant to the shared rescale).
                dpg_term = -actor_Q.mean() / frontier_norm
                coef_now = self.args.search_coef * min(
                    1.0, self.training_steps / max(1, self.args.search_ramp_steps)
                )
                search_term = coef_now * search_mse
                # grad-ratio guardrail (v40): refresh the cached attenuation at the hard-update
                # cadence; fires only above search_ceiling, inert when the term is small.
                # (Refresh actually lands every lcm(target_update_rate, policy_freq) steps —
                # equal to target_update_rate at the 250/2 defaults.)
                if self.training_steps % self.args.target_update_rate == 0:
                    g_dpg = torch.autograd.grad(dpg_term, list(self.actor.parameters()),
                                                retain_graph=True, allow_unused=True)
                    g_srch = torch.autograd.grad(search_term, list(self.actor.parameters()),
                                                 retain_graph=True, allow_unused=True)

                    def _norm(gs):
                        flats = [g.reshape(-1) for g in gs if g is not None]
                        return torch.cat(flats).norm().item() if flats else 0.0

                    ratio = _norm(g_srch) / (_norm(g_dpg) + 1e-8)
                    self.search_atten = min(1.0, self.args.search_ceiling / max(ratio, 1e-8))
                    self.search_stats["search_grad_ratio"] = ratio
                actor_loss = dpg_term + self.search_atten * search_term

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.training_steps % 500 == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)

        # hard target/fixed-encoder updates + target clip range snapshot
        if self.training_steps % self.args.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        # losses are logged against training_steps: with checkpointing, training happens in
        # episode-boundary bursts, so global_step would clump thousands of points at one x value
        if self.training_steps % 500 == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", Q.mean().item(), self.training_steps)
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            self.writer.add_scalar("debug/huber_linear_frac", self.huber_linear_frac, self.training_steps)
            if self.args.train_search:
                for tag, val in self.search_stats.items():
                    self.writer.add_scalar(f"debug/{tag}", val, self.training_steps)
                self.writer.add_scalar("debug/search_atten", self.search_atten, self.training_steps)
            if self.args.act_search_k > 0:
                self.writer.add_scalar("debug/act_search_gap", self.act_gap_ema, self.training_steps)

    # if using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # end assessment of the current policy early: it already lost to the checkpoint
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    # batch training at episode boundaries
    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(agent: TD7Agent, eval_env, eval_eps, use_checkpoint):
    returns = np.zeros(eval_eps)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(state), use_checkpoint=use_checkpoint, use_exploration=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            returns[ep] += reward
            done = terminated or truncated
    return returns.mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "TD7 requires num_envs=1 (1:1 train/env-step ratio and episodic checkpointing)"
    assert args.q_nstep >= 1
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
    action_dim = np.prod(envs.single_action_space.shape)
    max_action = float(envs.single_action_space.high[0])

    agent = TD7Agent(state_dim, action_dim, max_action, args, device, writer)
    nstep_acc = NStepAccumulator(args.q_nstep, args.gamma)

    start_time = time.time()
    allow_train = False

    obs, _ = envs.reset(seed=args.seed)
    eval_seeded = False
    # total_timesteps + 1 so the final evaluation at exactly total_timesteps fires (as in the original)
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
            actions = agent.select_action(np.array(obs[0]))[None]
        else:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # n-step accumulator -> replay buffer; handle `final_observation`. The original
        # bootstraps at the timeout step even when the env truly terminated there, hence
        # `and not truncations[0]` (terminated_q). Windows never cross a boundary; all
        # pending windows flush at the boundary BEFORE the training burst below.
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        boundary = bool(terminations[0] or truncations[0])
        terminated_q = float(terminations[0] and not truncations[0])
        for s, a, n1, ns, R, disc in nstep_acc.push(
            obs[0], actions[0], float(rewards[0]), real_next_obs, terminated_q, boundary
        ):
            agent.replay_buffer.add(s, a, n1, ns, R, disc)

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
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "encoder": agent.encoder.state_dict(),
                "critic": agent.critic.state_dict(),
                "checkpoint_actor": agent.checkpoint_actor.state_dict(),
                "checkpoint_encoder": agent.checkpoint_encoder.state_dict(),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()
