"""How accurate is REPPO's Q critic where its actor reads it: the action gradient and the within-state ranking?

    python scripts/reppo_critic_oracle.py CHECKPOINT.pt [CHECKPOINT.pt ...] [--out results.json]

REPPO's actor climbs dQ/da of its learned critic (pathwise), so what matters is the critic's action gradient
and its ranking of actions within a state, more than its level. This measures all three against the truth of
the frozen snapshot policy, on HalfCheetah-v4 only. It reuses scripts/awr_oracle_advantage.py's simulator
(gymnasium's model stepped by a threaded C kernel over restorable (qpos, qvel, qacc_warmstart) states,
bitwise-validated against gymnasium) and its counter-based hash RNG, and scripts/landscape.py's reppo_adapter
to rebuild the trainer's actor, critic and entropy term (--trainer, default the v60 noise fork).

Definitions, as the trainer (cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py):
- Rewards are raw * reward_scale (no reward normalization); observations use the frozen per-env
  VectorObsNorm rows, each state keeping its env's row.
- The critic's target is soft: Q(s, a) = r(s, a) + gamma E V'(s'), V'(s) = E_pi[Q(s, a') + b(s, a')] with the
  entropy bonus b = -T . log pi(a'|s) (per-dimension log-density of the squashed sample, T the snapshot's
  temperature). The bonus enters at every next state, never at s itself. The hard Q drops it.
- Monte Carlo estimate from (s, a): sum_{t<H} gamma^t r_t + sum_{1<=t<H} gamma^t b(s_t, a_t)
  + gamma^H V'_critic(s_H), the realized bonus being unbiased for the entropy. The hard return bootstraps
  E_pi Q_critic (the trainer's hard target does the same). V' averages --bootstrap-samples policy actions.
- Time limit. HalfCheetah never terminates and the critic does not see time, so the trainer's fixed point
  (its targets cut at the 1000-step limit and bootstrap V'(s_T)) is the infinite-horizon value. The primary
  truth ("soft") runs through the limit; "soft_limit" / "hard_limit" stop there and bootstrap V'_critic,
  exactly as the trainer's targets (their critic dependence is visible in the bootstrap weights).

Data: --states states drawn from every env lane's trajectory, started at the trainer's staggered phase
offsets, under the stochastic policy. Per state: the sampled action a0, K fresh actions a_k ~ pi, and
central-difference pairs along J Haar-random orthonormal directions u_j for each step h, in action space
(a0 +- h u_j, h shrunk per pair so |a| <= 1 - MARGIN; pairs shrunk below MIN_STEP_FRAC h are dropped) and in
the actor's pre-tanh coordinates (tanh(pre0 +- h u_j), where the reparameterized gradient flows). Each action
gets M rollouts of H steps; rollout m of every action of a state draws the same future policy noise (common
random numbers: Gaussian noise by inverse CDF of hash(state, m, t, dim)); a duplicate of a0 checks it. SEs
come from the spread over m. HalfCheetah is chaotic: a 1e-8 action change decorrelates CRN branches within
~100 steps, so derivative noise is ~ sd(Q) / h at any h and only aggregates over states resolve it.

Report per truth variant:
(a) level: Q_critic(s, a0) - Q_true bias and sd (rollout noise removed) against the between-state sd of Q_true.
(b) within-state: correlation of the critic's A(s, a_k) with the true one over the K fresh actions (truth with
    the leave-one-out baseline; pooled over states, raw and noise-corrected), the true within-state sd against
    the critic's, and the true advantage of the critic's favourite of the K actions.
(c) action gradient, true CRN central differences d against the critic's u_j . grad Q at a0 (autograd) in the
    J-dim projections. Unbiased at any noise level, with SEs: the slope of truth on critic (1 correct,
    0 uninformative) and the ascent ratio (true rate of improvement along the critic's unit gradient over
    the rate it claims). Noise-limited: pooled uncentered correlation (raw and noise-corrected), norm ratio,
    per-state cosine (median, with its rollout-noise ceiling and the split-half truth self-cosine), sign
    agreement, the critic's own finite difference, and linearity across h (cross-half).
(d) the one-step Bellman residual r + gamma V'_critic(s') - Q_critic(s, a) against the true error
    Q_critic - Q_true, at a0 and within states.
Horizon sweep: the same with Q bootstrapped by the critic at t in --checkpoints (t = 1 is the one-step
lookahead, whose gradient is nearly noise-free): low noise but critic-contaminated at small t, the truth at H.

Run it through mlq (CUDA). Physics dominates: N (2 + K + 4 J |h|) M H env steps.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awr_oracle_advantage as awr  # noqa: E402  (also puts the repo root on sys.path)
import landscape  # noqa: E402

from cleanrl.shared.runtime import configure_runtime  # noqa: E402
from cleanrl.shared.staggered_envs import compute_phase_offsets  # noqa: E402

ENV_ID, TIME_LIMIT = awr.ENV_ID, awr.TIME_LIMIT
DEFAULT_TRAINER = Path(__file__).resolve().parents[1] / "cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py"
MARGIN = 1e-3  # action-space pairs stay within |a| <= 1 - MARGIN
MIN_STEP_FRAC = 0.25  # action-space pairs shrunk below this fraction of h are dropped
SPACES = ("action", "pre")
VARIANTS = ("soft", "hard", "soft_limit", "hard_limit")
# Counter-based random streams: every draw is a hash of (stream, ...), so any branch can recreate it.
COLLECT, FRESH, DIRECTIONS, BRANCH, TAIL, LIMIT, VALIDATE, TEST = range(11, 19)


# ---------------------------------------------------------------- counter-based Gaussian noise

def hash_normals(base, count):
    """(..., count) standard normals, draw j = Phi^-1(u) with u from all 32 bits of hash(base, j).

    float64 inverse CDF: u lies in [2^-33, 1 - 2^-33], so every draw is finite (|z| < 6.3)."""
    bits = awr.mix32(awr.fold(base[..., None], torch.arange(count, device=base.device)))
    u = (bits.double() + 0.5) * 2.0 ** -32
    return (math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)).float()


def orthonormal_directions(base, dims, count):
    """(..., count, dims) orthonormal rows, Haar-distributed: QR of a hashed Gaussian matrix, signs fixed."""
    gaussian = hash_normals(base, dims * dims).double().view(*base.shape, dims, dims)
    q, r = torch.linalg.qr(gaussian)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    return q.transpose(-1, -2)[..., :count, :].float()


def validate_normals(device, seed_hash):
    """KS-test the hashed normals and check they are uncorrelated across dimensions and counters."""
    count = 200_000
    z = hash_normals(awr.stream_key(seed_hash, TEST, torch.arange(count, device=device)), 6).double().cpu().numpy()
    ks = stats.kstest(z[:, 0], "norm")
    corr = np.corrcoef(z.T)
    report = {"ks": float(ks.statistic), "p": float(ks.pvalue), "mean": float(z.mean()), "var": float(z.var()),
              "max_cross_dim_corr": float(np.abs(corr - np.eye(6)).max()),
              "lag1_corr": float(np.corrcoef(z[:-1, 0], z[1:, 0])[0, 1])}
    assert ks.pvalue > 1e-4 and abs(report["var"] - 1.0) < 0.01, f"hashed normals are not standard: {report}"
    assert report["max_cross_dim_corr"] < 0.02 and abs(report["lag1_corr"]) < 0.02, f"hashed normals correlate: {report}"
    return report


# ---------------------------------------------------------------- return accounting

def account_reward(totals, t, gamma, reward, at_limit):
    """The branch has just reached s_t: add r_{t-1} with gamma^(t-1), in place.

    totals (4, n): soft and hard returns through the time limit, then both frozen at it. The frozen copy is
    taken here, before s_t's bonus: the trainer's cut bootstraps V'(s_t), which contains it. Any bootstrap at
    s_t (checkpoints, horizon) likewise reads totals between account_reward and account_bonus."""
    totals[:2] += gamma ** (t - 1) * reward
    totals[2:] = torch.where(at_limit, totals[:2], totals[2:])


def account_bonus(totals, t, gamma, bonus):
    """The entropy bonus b(s_t, a_t) = -T . log pi(a_t|s_t) of the action sampled at s_t, with gamma^t."""
    totals[0] += gamma ** t * bonus


def bootstrap(totals, to_limit, horizon, gamma, tail_soft, tail_hard, limit_soft, limit_hard):
    """Q estimates (4, n) in VARIANTS order, and each branch's time-limit bootstrap weight.

    to_limit: steps from s_0 to the time-limit state. Through the limit, every branch bootstraps
    gamma^H V'(s_H); stopped at it, branches that reach it within H bootstrap gamma^to_limit V'(s_limit)."""
    tail = gamma ** horizon
    soft, hard = totals[0] + tail * tail_soft, totals[1] + tail * tail_hard
    cut = to_limit <= horizon
    weight = torch.where(cut, gamma ** to_limit.double(), tail)
    soft_limit = torch.where(cut, totals[2] + weight * limit_soft, soft)
    hard_limit = torch.where(cut, totals[3] + weight * limit_hard, hard)
    return torch.stack((soft, hard, soft_limit, hard_limit)), weight


# ---------------------------------------------------------------- finite differences

def difference_actions(a0, pre0, directions, steps):
    """Central-difference pairs (n, 2 spaces, |h|, J, 2 signs, dims), their steps (n, 2, |h|, J), validity.

    action space: a0 +- h u_j with h shrunk per pair so every coordinate stays within 1 - MARGIN;
    pre space: tanh(pre0 +- h u_j), always interior."""
    sign = torch.tensor([1.0, -1.0], device=a0.device)[:, None]
    u = directions[:, None, :, None, :]  # (n, 1, J, 1, dims)
    room = ((1.0 - MARGIN - a0.abs())[:, None, :] / directions.abs().clamp_min(1e-12)).amin(-1)  # (n, J)
    h_action = torch.minimum(steps[None, :, None], room[:, None, :]).clamp_min(0.0)  # (n, |h|, J)
    action = a0[:, None, None, None, :] + sign * h_action[..., None, None] * u
    pre = torch.tanh(pre0[:, None, None, None, :] + sign * steps[None, :, None, None, None] * u)
    h = torch.stack((h_action, steps[None, :, None].expand_as(h_action)), 1)
    valid = torch.stack((h_action >= MIN_STEP_FRAC * steps[None, :, None], torch.ones_like(h_action, dtype=torch.bool)), 1)
    return torch.stack((action, pre), 1), h, valid


def directional_derivatives(plus, minus, h):
    """Central differences (Q(+h) - Q(-h)) / 2h; dropped pairs (h = 0) give nan."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (plus - minus) / (2.0 * h)


def leave_one_out(q, axis=1):
    """A(s, a_k) = Q_k - mean_{j != k} Q_j = K / (K - 1) (Q_k - mean_j Q_j): unbiased for Q(s, a_k) - V(s)."""
    count = q.shape[axis]
    return count / (count - 1) * (q - q.mean(axis, keepdims=True))


# ---------------------------------------------------------------- frozen policy and critic

class Frozen:
    """The snapshot's tanh-Gaussian actor, Q critic and entropy term, with frozen per-env normalization."""

    def __init__(self, snapshot, trainer, physics, device, seed_hash):
        adapter, env_id = landscape.reppo_adapter(snapshot, trainer, device)
        if env_id != ENV_ID:
            raise ValueError(f"the simulator is {ENV_ID}-only; snapshot is {env_id}")
        self.policy = adapter.module.requires_grad_(False).eval()
        self.critic, self.bonus_of = adapter.critic, adapter.regularizer
        self.physics, self.device, self.seed_hash = physics, device, seed_hash
        self.dims = adapter.action_dim
        args = snapshot["args"]
        self.gamma, self.reward_scale, self.obs_clip = float(args["gamma"]), float(args["reward_scale"]), adapter.obs_clip
        as64 = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float64, device=device)  # noqa: E731
        self.obs_mean = as64(snapshot["obs_means"])
        self.obs_scale = torch.sqrt(as64(snapshot["obs_variances"]) + float(snapshot.get("obs_epsilon", 1e-8)))
        self.lanes = self.obs_mean.shape[0]
        self.temperature = [float(value) for value in snapshot["temperature"]]

    def observe(self, state, rows):
        """HalfCheetah obs (qpos[1:], qvel), normalized with the rows' frozen statistics, as VectorObsNorm."""
        obs = state[:, 1:self.physics.nq + self.physics.nv]
        return ((obs - self.obs_mean[rows]) / self.obs_scale[rows]).clamp(-self.obs_clip, self.obs_clip).float()

    def raw_reward(self, x_after, x_before, action):
        forward = awr.FORWARD_REWARD_WEIGHT * ((x_after - x_before) / self.physics.dt)
        return forward - awr.CTRL_COST_WEIGHT * action.double().square().sum(-1)

    def moments(self, obs):
        return self.policy(obs).chunk(2, dim=-1)

    def act(self, obs, noise):
        """tanh(mean + std noise) and its entropy bonus -T . log pi."""
        outputs = self.policy(obs)
        mean, std = outputs.chunk(2, dim=-1)
        return torch.tanh(mean + std * noise), self.bonus_of(outputs, noise, 1.0)

    def soft_value(self, obs, keys, samples, chunk=8192):
        """V'(s) = mean_k [Q(s, a_k) + b(s, a_k)] and E_pi Q = mean_k Q(s, a_k) over hashed policy samples."""
        soft, hard = [], []
        for start in range(0, obs.shape[0], chunk):
            part = obs[start:start + chunk, None].expand(-1, samples, -1)
            noise = hash_normals(awr.fold(keys[start:start + chunk, None], torch.arange(samples, device=obs.device)),
                                 self.dims)
            action, bonus = self.act(part, noise)
            q = self.critic(part, action)
            soft.append((q + bonus).mean(1).double())
            hard.append(q.mean(1).double())
        return torch.cat(soft), torch.cat(hard)

    def action_gradient(self, obs, action):
        with torch.enable_grad():
            action = action.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(self.critic(obs, action).sum(), action)
        return grad


# ---------------------------------------------------------------- branch rollouts

class BranchGroup:
    """All (state, action, rollout) branches of one group of states, stepped in lockstep."""

    def __init__(self, frozen, part, actions, rollouts, horizon, checkpoints, samples):
        count, per_state, dims = actions.shape
        device, physics = frozen.device, frozen.physics
        self.frozen, self.horizon, self.shape = frozen, horizon, (count, per_state, rollouts)
        self.checkpoints, self.samples = checkpoints, samples
        per = per_state * rollouts
        size = count * per
        pinned = device.type == "cuda"
        self.state = torch.empty((size, physics.width), dtype=torch.float64, pin_memory=pinned)
        self.state_np = self.state.numpy()
        self.state_np[:] = np.repeat(part["state"], per, axis=0)
        self.state_gpu = torch.empty(self.state.shape, dtype=torch.float64, device=device)
        self.ctrl = torch.empty((size, dims), dtype=torch.float32, pin_memory=pinned)
        self.ctrl_np = self.ctrl.numpy()
        expand = lambda values: torch.as_tensor(values, device=device).repeat_interleave(per)  # noqa: E731
        self.rows, self.sids = expand(part["row"]), expand(part["sid"])
        self.to_limit = expand(TIME_LIMIT - part["age"])
        self.rollout = torch.arange(rollouts, device=device).repeat(count * per_state)
        self.action = actions.reshape(-1, dims).repeat_interleave(rollouts, 0)
        self.ctrl.copy_(self.action)
        self.x_before = expand(part["state"][:, 0])
        self.totals = torch.zeros((4, size), dtype=torch.float64, device=device)
        self.limit_obs = torch.zeros((size, physics.nq - 1 + physics.nv), device=device)
        self.checks, self.tail, self.first_reward = {}, None, None

    def advance(self, t):
        frozen = self.frozen
        self.state_gpu.copy_(self.state, non_blocking=True)
        x = self.state_gpu[:, 0]
        reward = frozen.reward_scale * frozen.raw_reward(x, self.x_before, self.action)
        obs = frozen.observe(self.state_gpu, self.rows)
        at_limit = self.to_limit == t
        account_reward(self.totals, t, frozen.gamma, reward, at_limit)
        self.limit_obs = torch.where(at_limit[:, None], obs, self.limit_obs)
        if t == 1:  # r(s, a) and s' are deterministic given (s, a): every rollout must agree bitwise
            first = reward.view(self.shape)
            if not torch.equal(first, first[..., :1].expand_as(first)):
                raise AssertionError("the first transition differs across rollouts of one (state, action)")
            self.first_reward = first[..., 0].double().cpu().numpy()
        if t in self.checkpoints:
            # Q bootstrapped by the critic at s_t (keys shared by every action of a state and rollout).
            keys = awr.stream_key(frozen.seed_hash, TAIL, self.sids, self.rollout, t)
            soft, hard = frozen.soft_value(obs, keys, self.samples)
            self.checks[t] = torch.stack((self.totals[0] + frozen.gamma ** t * soft,
                                          self.totals[1] + frozen.gamma ** t * hard))
            if t == self.horizon:
                self.tail = soft, hard
        if t < self.horizon:
            noise = hash_normals(awr.stream_key(frozen.seed_hash, BRANCH, self.sids, self.rollout, t), frozen.dims)
            action, bonus = frozen.act(obs, noise)
            account_bonus(self.totals, t, frozen.gamma, bonus)
            self.action, self.x_before = action, x.clone()  # x is a view of the buffer the next step overwrites
            self.ctrl.copy_(action, non_blocking=True)
        if frozen.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    def finish(self):
        frozen, (count, per_state, rollouts) = self.frozen, self.shape
        tail_soft, tail_hard = self.tail
        limit_soft = torch.zeros_like(tail_soft)
        limit_hard = torch.zeros_like(tail_hard)
        cut = torch.nonzero(self.to_limit <= self.horizon).flatten()
        if cut.numel():
            keys = awr.stream_key(frozen.seed_hash, LIMIT, self.sids[cut], self.rollout[cut])
            limit_soft[cut], limit_hard[cut] = frozen.soft_value(self.limit_obs[cut], keys, self.samples)
        q, weight = bootstrap(self.totals, self.to_limit, self.horizon, frozen.gamma, tail_soft, tail_hard,
                              limit_soft, limit_hard)
        view = lambda value: value.view(-1, count, per_state, rollouts).cpu().numpy()  # noqa: E731
        return {"q": view(q), "checks": np.stack([view(self.checks[t]) for t in self.checkpoints], 1),
                "limit_weight": view(weight)[0, :, 0, 0], "tail": frozen.gamma ** self.horizon * view(tail_soft)[0],
                "first_reward": self.first_reward}


@torch.no_grad()
def branch_returns(frozen, data, actions, cli, pool):
    """Every group's finish() fields, concatenated over states. Two groups are in flight: one group's GPU
    step overlaps the other's physics (ctypes drops the GIL)."""
    count = len(data["sid"])
    starts = list(range(0, count, cli.group_states))
    out = []
    begin = time.perf_counter()
    for index in range(0, len(starts), 2):
        groups = [BranchGroup(frozen, {key: value[s:s + cli.group_states] for key, value in data.items()},
                              actions[s:s + cli.group_states], cli.rollouts, cli.horizon, cli.checkpoints,
                              cli.bootstrap_samples)
                  for s in starts[index:index + 2]]
        pending, owner = None, None
        for t in range(cli.horizon + 1):
            for group in groups:
                if owner is group:
                    pending.result()
                    pending = owner = None
                if t > 0:
                    group.advance(t)
                if t < cli.horizon:
                    if pending is not None:
                        pending.result()
                    pending, owner = pool.submit(frozen.physics.step, group.state_np, group.ctrl_np), group
        if pending is not None:
            pending.result()
        out.extend(group.finish() for group in groups)
        done = min(count, starts[min(index + 2, len(starts)) - 1] + cli.group_states)
        elapsed = time.perf_counter() - begin
        steps = done * actions.shape[1] * cli.rollouts * cli.horizon
        print(f"  branches: {done}/{count} states, {elapsed:.0f}s, {steps / elapsed / 1e6:.2f}M env steps/s, "
              f"eta {elapsed * (count - done) / done:.0f}s", flush=True)
    axis = {"q": 1, "checks": 2, "limit_weight": 0, "tail": 0, "first_reward": 0}  # the states axis
    return {key: np.concatenate([part[key] for part in out], axis=axis[key]) for key in out[0]}


# ---------------------------------------------------------------- data

@torch.no_grad()
def collect(frozen, count, steps, seed):
    """`count` states drawn uniformly from `steps` steps of every env lane, started at the trainer's staggered
    phase offsets, under the stochastic policy. Each lane keeps its env's normalization row."""
    physics, device, lanes = frozen.physics, frozen.device, frozen.lanes
    state = physics.reset(lanes)
    age = np.zeros(lanes, dtype=np.int64)
    rows = torch.arange(lanes, device=device)
    clock = 0

    def act():
        nonlocal clock
        mean, std = frozen.moments(frozen.observe(torch.from_numpy(state).to(device), rows))
        pre = mean + std * hash_normals(awr.stream_key(frozen.seed_hash, COLLECT, rows, clock), frozen.dims)
        clock += 1
        return pre, torch.tanh(pre)

    offsets = compute_phase_offsets(lanes, TIME_LIMIT, seed)
    for step in range(int(offsets.max())):
        stepped = state.copy()
        physics.step(stepped, act()[1].cpu().numpy())
        active = offsets > step
        state[active], age[active] = stepped[active], age[active] + 1
    pool = {"state": np.empty((steps, lanes, physics.width)), "age": np.empty((steps, lanes), dtype=np.int64),
            "pre": np.empty((steps, lanes, frozen.dims), dtype=np.float32),
            "action": np.empty((steps, lanes, frozen.dims), dtype=np.float32)}
    for step in range(steps):
        pre, action = act()
        pool["state"][step], pool["age"][step] = state, age
        pool["pre"][step], pool["action"][step] = pre.cpu().numpy(), action.cpu().numpy()
        physics.step(state, pool["action"][step])
        age += 1
        done = np.flatnonzero(age >= TIME_LIMIT)
        if done.size:
            state[done], age[done] = physics.reset(done.size), 0
    pick = np.sort(np.random.default_rng(seed).choice(steps * lanes, count, replace=False))
    data = {key: value.reshape(steps * lanes, *value.shape[2:])[pick] for key, value in pool.items()}
    data["row"], data["sid"] = pick % lanes, pick
    return data


@torch.no_grad()
def action_set(frozen, data, cli):
    """(n, A, dims) actions per state: a0, K fresh, the difference pairs, then a0 again (a CRN determinism
    check: its returns must equal a0's); plus the pairs' layout."""
    device = frozen.device
    sids = torch.as_tensor(data["sid"], device=device)
    obs = frozen.observe(torch.as_tensor(data["state"], device=device), torch.as_tensor(data["row"], device=device))
    a0 = torch.as_tensor(data["action"], device=device)
    pre0 = torch.as_tensor(data["pre"], device=device)
    mean, std = frozen.moments(obs)
    noise = hash_normals(awr.fold(awr.stream_key(frozen.seed_hash, FRESH, sids)[:, None],
                                  torch.arange(cli.fresh, device=device)), frozen.dims)
    fresh = torch.tanh(mean[:, None] + std[:, None] * noise)
    directions = orthonormal_directions(awr.stream_key(frozen.seed_hash, DIRECTIONS, sids), frozen.dims, cli.directions)
    steps = torch.tensor(cli.steps, device=device)
    pairs, h, valid = difference_actions(a0, pre0, directions, steps)
    actions = torch.cat([a0[:, None], fresh, pairs.flatten(1, 4), a0[:, None]], 1)
    return obs, actions, {"directions": directions, "h": h, "valid": valid, "a0": a0}


@torch.no_grad()
def critic_readout(frozen, obs, actions, layout, chunk=4096):
    """Q_critic at every action (n, A) and its projected action gradients at a0 (n, 2 spaces, J).

    Pre space: d/dh Q(tanh(pre0 + h u)) = u . (grad_a Q * (1 - a0^2))."""
    q = torch.cat([frozen.critic(obs[s:s + chunk, None].expand(-1, actions.shape[1], -1), actions[s:s + chunk])
                   for s in range(0, obs.shape[0], chunk)])
    a0 = layout["a0"]
    grad = frozen.action_gradient(obs, a0)
    gradients = torch.stack((grad, grad * (1.0 - a0.square())), 1)  # (n, 2, dims)
    projected = torch.einsum("njd,nsd->nsj", layout["directions"], gradients)
    return q.double().cpu().numpy(), projected.double().cpu().numpy(), grad.double().cpu().numpy()


# ---------------------------------------------------------------- statistics

TINY = 1e-30
SIGNAL_FLOOR = 0.05  # |d|^2-based statistics need the truth's signal to be at least this share of sum d_hat^2


def _float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def disattenuate(corr, reliability):
    """corr / sqrt(reliability of the noisy side); None when the reliability is below SIGNAL_FLOOR."""
    return _float(corr / math.sqrt(reliability)) if reliability >= SIGNAL_FLOOR else None


def cosines(x, y):
    return (x * y).sum(-1) / np.maximum(np.sqrt((x * x).sum(-1) * (y * y).sum(-1)), TINY)


def level_stats(critic, samples):
    """Critic (n,) against Monte Carlo samples (n, M) of the truth, rollout noise removed from the spreads."""
    truth = samples.mean(-1)
    noise = (samples.var(-1, ddof=1) / samples.shape[-1]).mean()
    error = critic - truth
    between = max(truth.var() - noise, 0.0)
    corr = np.corrcoef(critic, truth)[0, 1]
    return {"truth_mean": _float(truth.mean()), "critic_mean": _float(critic.mean()), "bias": _float(error.mean()),
            "error_sd": _float(math.sqrt(max(error.var() - noise, 0.0))), "between_sd": _float(math.sqrt(between)),
            "error_sd_over_between": _float(math.sqrt(max(error.var() - noise, 0.0) / max(between, TINY))),
            "corr": _float(corr), "corr_corrected": disattenuate(corr, between / max(truth.var(), TINY)),
            "se_rms": _float(math.sqrt(noise))}


def within_stats(critic, samples):
    """Critic (n, K) against samples (n, K, M) of Q at K policy actions per state, CRN-paired over M."""
    count, rollouts = samples.shape[1], samples.shape[2]
    deviation_m = samples - samples.mean(1, keepdims=True)
    deviation = deviation_m.mean(-1)
    noise = deviation_m.var(-1, ddof=1) / rollouts
    truth = leave_one_out(samples.mean(-1))
    ours = critic - critic.mean(1, keepdims=True)
    critic_var = (ours ** 2).sum(1).mean() / (count - 1)
    corr = (ours * truth).sum() / math.sqrt(max((ours ** 2).sum() * (truth ** 2).sum(), TINY))
    reliability = 1.0 - noise.sum() / max((deviation ** 2).sum(), TINY)
    # The true within-state sd (rollout noise removed); undefined when the noise swamps it.
    true_sd = math.sqrt(((deviation ** 2).sum(1) - noise.sum(1)).mean() / (count - 1)) if reliability >= SIGNAL_FLOOR else None
    # True advantage of the critic's favourite among the K actions (0 for a random pick), in Q units and true sds.
    # Centered on all K (unbiased for V whatever is picked); the leave-one-out baseline would be biased here,
    # since the others are the actions the critic ranked below the pick.
    pick = deviation[np.arange(len(deviation)), ours.argmax(1)]
    pick_se = pick.std(ddof=1) / math.sqrt(len(pick))
    in_sd = lambda value: None if true_sd is None else _float(value / true_sd)  # noqa: E731
    return {"true_sd": true_sd and _float(true_sd), "critic_sd": _float(math.sqrt(critic_var)),
            "sd_ratio": in_sd(math.sqrt(critic_var)), "corr": _float(corr),
            "corr_corrected": disattenuate(corr, reliability), "reliability": _float(reliability),
            "per_state_corr_median": _float(np.median(cosines(ours, truth))),
            "critic_pick_gain": _float(pick.mean()), "critic_pick_gain_se": _float(pick_se),
            "critic_pick_gain_sd": in_sd(pick.mean()), "critic_pick_gain_sd_se": in_sd(pick_se)}


def gradient_stats(samples, critic, critic_fd):
    """True directional-derivative samples (n, J, M) against the critic's autograd (n, J) and FD (n, J).

    Chaos decorrelates CRN branches within ~50 steps, so each derivative carries noise ~ sd(Q) / h that no
    step size removes. The statistics linear in the truth are unbiased at any noise level and get SEs from
    the m-spread (per state, so the directions' shared noise is accounted): slope_true_on_critic (1 for a
    correct gradient, 0 for an uninformative one) and ascent_ratio, the true rate of improvement along the
    critic's unit gradient over the rate the critic claims. Those needing |d|^2 (noise-corrected correlation,
    norm ratio) are None when the noise-corrected signal is below SIGNAL_FLOOR of sum d_hat^2."""
    if len(samples) == 0:
        return {"states": 0}
    rollouts = samples.shape[-1]
    truth = samples.mean(-1)
    noise = samples.var(-1, ddof=1) / rollouts
    half_a, half_b = samples[..., :rollouts // 2].mean(-1), samples[..., rollouts // 2:].mean(-1)
    total = (truth ** 2).sum()
    signal_frac = 1.0 - noise.sum() / max(total, TINY)
    signal = total - noise.sum() if signal_frac >= SIGNAL_FLOOR else None
    dot, critic_sq, fd_sq = (truth * critic).sum(), (critic ** 2).sum(), (critic_fd ** 2).sum()
    norms = np.maximum(np.linalg.norm(critic, axis=-1), TINY)
    along = np.einsum("njm,nj->nm", samples, critic / norms[:, None])  # true rate along the critic's direction
    projected = np.einsum("njm,nj->nm", samples, critic)
    sum_se = lambda values: math.sqrt((values.var(-1, ddof=1) / rollouts).sum())  # noqa: E731
    significant = np.abs(truth) > 2.0 * np.sqrt(noise)
    agree = np.sign(truth) == np.sign(critic)
    ceiling = np.sqrt(np.clip(1.0 - noise.sum(-1) / np.maximum((truth ** 2).sum(-1), TINY), 0.0, 1.0))
    corrected = lambda value: None if signal is None else _float(value)  # noqa: E731
    return {"states": len(samples),
            "slope_true_on_critic": _float(dot / max(critic_sq, TINY)),
            "slope_se": _float(sum_se(projected) / max(critic_sq, TINY)),
            "ascent_ratio": _float(along.mean(-1).sum() / norms.sum()), "ascent_se": _float(sum_se(along) / norms.sum()),
            "true_rate_along_critic": _float(along.mean()), "critic_rate": _float(norms.mean()),
            "signal_frac": _float(signal_frac),
            "pooled_corr": _float(dot / math.sqrt(max(total * critic_sq, TINY))),
            "pooled_corr_corrected": corrected(signal and dot / math.sqrt(max(signal * critic_sq, TINY))),
            "norm_ratio_critic_over_true": corrected(signal and math.sqrt(critic_sq / signal)),
            "critic_fd_corr_corrected": corrected(signal and (truth * critic_fd).sum() / math.sqrt(max(signal * fd_sq, TINY))),
            "critic_fd_slope": _float((truth * critic_fd).sum() / max(fd_sq, TINY)),
            "critic_fd_vs_autograd_cos_median": _float(np.median(cosines(critic_fd, critic))),
            "cos_median": _float(np.median(cosines(truth, critic))), "cos_ceiling_median": _float(np.median(ceiling)),
            "split_half_cos_median": _float(np.median(cosines(half_a, half_b))),
            "sign_agreement": _float(agree.mean()),
            "sign_agreement_significant": _float(agree[significant].mean()) if significant.any() else None,
            "significant_frac": _float(significant.mean()),
            "true_rms": corrected(signal and math.sqrt(signal / truth.size)), "se_rms": _float(math.sqrt(noise.mean())),
            "critic_rms": _float(math.sqrt(critic_sq / truth.size))}


def linearity_stats(small, large):
    """Do two step sizes' true derivatives (n, J, M) agree? Cross-half products are free of the noise the two
    share through their common rollouts; None when either signal is below SIGNAL_FLOOR."""
    if len(small) == 0:
        return {"states": 0}
    half = small.shape[-1] // 2
    a1, b1, a2, b2 = small[..., :half].mean(-1), small[..., half:].mean(-1), large[..., :half].mean(-1), large[..., half:].mean(-1)
    cross = 0.5 * ((a1 * b2).sum() + (b1 * a2).sum())

    def signal(samples):
        total = (samples.mean(-1) ** 2).sum()
        value = total - (samples.var(-1, ddof=1) / samples.shape[-1]).sum()
        return value if value >= SIGNAL_FLOOR * total else None

    small_signal, large_signal = signal(small), signal(large)
    if small_signal is None or large_signal is None:
        return {"states": len(small), "corr_corrected": None, "norm_ratio_small_over_large": None,
                "slope_small_on_large": None}
    return {"states": len(small), "corr_corrected": _float(cross / math.sqrt(small_signal * large_signal)),
            "norm_ratio_small_over_large": _float(math.sqrt(small_signal / large_signal)),
            "slope_small_on_large": _float(cross / large_signal)}


def residual_stats(residual, critic, samples, fresh_residual, fresh_critic, fresh_samples):
    """One-step Bellman residual against the true critic error, at a0 (n,) and within states (n, K)."""
    error_m = critic[:, None] - samples
    error = error_m.mean(-1)
    noise = (error_m.var(-1, ddof=1) / samples.shape[-1]).mean()
    corr = np.corrcoef(residual, -error)[0, 1]
    deviation_m = fresh_samples - fresh_samples.mean(1, keepdims=True)
    advantage_error = (fresh_critic - fresh_critic.mean(1, keepdims=True)) - deviation_m.mean(-1)
    advantage_noise = (deviation_m.var(-1, ddof=1) / fresh_samples.shape[-1]).sum()
    centered = fresh_residual - fresh_residual.mean(1, keepdims=True)
    within = (centered * -advantage_error).sum() / math.sqrt(max((centered ** 2).sum() * (advantage_error ** 2).sum(), TINY))
    within_reliability = 1.0 - advantage_noise / max((advantage_error ** 2).sum(), TINY)
    return {"residual_mean": _float(residual.mean()), "residual_sd": _float(residual.std()),
            "error_sd": _float(math.sqrt(max(error.var() - noise, 0.0))),
            "corr_residual_neg_error": _float(corr),
            "corr_corrected": disattenuate(corr, 1.0 - noise / max(error.var(), TINY)),
            "slope_neg_error_on_residual": _float(np.cov(residual, -error)[0, 1] / max(residual.var(ddof=1), TINY)),
            "within_corr": _float(within), "within_corr_corrected": disattenuate(within, within_reliability)}


def pair_derivatives(values, h, fresh, steps):
    """Central differences of per-action values (n, A, *rest) over the pairs' layout: (n, 2, |h|, J, *rest)."""
    rest = values.shape[2:]
    pairs = values[:, 1 + fresh:-1].reshape(values.shape[0], len(SPACES), len(steps), -1, 2, *rest)
    step = h.reshape(h.shape + (1,) * len(rest))
    return directional_derivatives(pairs[:, :, :, :, 0], pairs[:, :, :, :, 1], step)


def model_based(q, one_step, reward, projected, h, valid, fresh, steps):
    """Would the actor do better reading its gradient off r(s, a) + gamma V'_critic(f(s, a)) through the true
    one-step dynamics f than off dQ_critic/da? The true derivatives (samples q at H) against each predictor.

    one_step: {name: (n, A)} one-step lookahead values (their V' draws are shared by every action of a state
    and rollout, so their differences are smooth); reward: (n, A) r(s, a) alone."""
    truth = pair_derivatives(q, h, fresh, steps)
    predicted = {name: pair_derivatives(values, h, fresh, steps) for name, values in one_step.items()}
    reward_fd = pair_derivatives(reward, h, fresh, steps)
    ok = valid.all(-1)
    out = {}
    for s, space in enumerate(SPACES):
        out[space] = {}
        for i, step in enumerate(steps):
            keep = ok[:, s, i]
            cell = {name: gradient_stats(truth[keep, s, i], value[keep, s, i], value[keep, s, i])
                    for name, value in predicted.items()}
            model = predicted["model_soft"][keep, s, i]
            cell["reward_share"] = _float((reward_fd[keep, s, i] ** 2).sum() / max((model ** 2).sum(), TINY)) if keep.any() else None
            cell["model_vs_critic_cos_median"] = _float(np.median(cosines(model, projected[keep, s]))) if keep.any() else None
            cell["model_vs_critic_slope"] = _float((model * projected[keep, s]).sum() / max((projected[keep, s] ** 2).sum(), TINY)) if keep.any() else None
            out[space][f"{step:g}"] = cell
    return out


def analyze(q, critic, projected, h, valid, residual, fresh, steps):
    """All statistics for one truth variant: q (n, A, M) samples, critic (n, A), projected (n, 2, J)."""
    base = 1 + fresh
    derivative = pair_derivatives(q, h, fresh, steps)  # (n, 2, |h|, J, M)
    critic_fd = pair_derivatives(critic, h, fresh, steps)  # (n, 2, |h|, J)
    ok = valid.all(-1)  # (n, 2, |h|): every direction of the state usable
    gradient, linearity = {}, {space: {} for space in SPACES}
    for s, space in enumerate(SPACES):
        gradient[space] = {f"{step:g}": gradient_stats(derivative[ok[:, s, i], s, i], projected[ok[:, s, i], s],
                                                       critic_fd[ok[:, s, i], s, i])
                           for i, step in enumerate(steps)}
        for i in range(len(steps) - 1):
            both = ok[:, s, i] & ok[:, s, -1]
            linearity[space][f"{steps[i]:g}/{steps[-1]:g}"] = linearity_stats(derivative[both, s, i], derivative[both, s, -1])
    fresh_q = q[:, 1:base]
    return {"level_a0": level_stats(critic[:, 0], q[:, 0]),
            "level_v": level_stats(critic[:, 1:base].mean(1), fresh_q.mean(1)),
            "within": within_stats(critic[:, 1:base], fresh_q), "gradient": gradient, "linearity": linearity,
            "residual": residual_stats(residual[:, 0], critic[:, 0], q[:, 0], residual[:, 1:base],
                                       critic[:, 1:base], fresh_q)}


# ---------------------------------------------------------------- validation

@torch.no_grad()
def episode_returns(frozen, episodes, seed):
    """Raw stochastic-policy episode returns in this simulator (hashed noise) and in gymnasium (torch noise),
    episode e normalized with row e % lanes in both."""
    physics, device = frozen.physics, frozen.device
    rows = torch.arange(episodes, device=device) % frozen.lanes
    state = physics.reset(episodes)
    ours = torch.zeros(episodes, dtype=torch.float64, device=device)
    for step in range(TIME_LIMIT):
        before = torch.from_numpy(state[:, 0]).to(device, copy=True)  # the step overwrites state in place
        mean, std = frozen.moments(frozen.observe(torch.from_numpy(state).to(device), rows))
        noise = hash_normals(awr.stream_key(frozen.seed_hash, VALIDATE, torch.arange(episodes, device=device), step),
                             frozen.dims)
        action = torch.tanh(mean + std * noise)
        physics.step(state, action.cpu().numpy())
        ours += frozen.raw_reward(torch.from_numpy(state[:, 0]).to(device), before, action)
    envs = [gym.make(ENV_ID) for _ in range(episodes)]
    raw = np.stack([env.reset(seed=seed + 10_000 + e)[0] for e, env in enumerate(envs)])
    generator = torch.Generator(device=device).manual_seed(seed)
    theirs = np.zeros(episodes)
    for step in range(TIME_LIMIT):
        full = np.zeros((episodes, physics.width))
        full[:, 1:1 + raw.shape[1]] = raw
        mean, std = frozen.moments(frozen.observe(torch.from_numpy(full).to(device), rows))
        action = torch.tanh(mean + std * torch.randn(mean.shape, device=device, generator=generator)).cpu().numpy()
        for e, env in enumerate(envs):
            raw[e], reward, terminated, truncated, _ = env.step(action[e])
            theirs[e] += reward
            assert not terminated and truncated == (step + 1 == TIME_LIMIT)
    for env in envs:
        env.close()
    ours = ours.cpu().numpy()
    z = (ours.mean() - theirs.mean()) / math.sqrt(ours.var(ddof=1) / episodes + theirs.var(ddof=1) / episodes)
    return {"simulator_mean": float(ours.mean()), "simulator_sd": float(ours.std()), "gymnasium_mean": float(theirs.mean()),
            "gymnasium_sd": float(theirs.std()), "episodes": episodes, "z": float(z)}


# ---------------------------------------------------------------- driver

def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def print_report(result, primary="soft"):
    header, variants = result["header"], result["variants"]
    print(f"step {header['global_step']}: N {header['states']}, K {header['fresh']}, J {header['directions']}, "
          f"M {header['rollouts']}, H {header['horizon']} (gamma^H {header['gamma_h']:.3f}), T {header['temperature'][0]:.4f}; "
          f"bootstrap at H: mean {result['bootstrap']['tail_mean']:.2f}, between-state sd "
          f"{result['bootstrap']['tail_between_sd']:.3f}; limit within H for {result['bootstrap']['limit_cut_frac']:.0%} "
          f"of states, weight {fmt(result['bootstrap']['limit_weight_mean'], '.3f')}", flush=True)
    for name in (primary, "hard", "soft_limit"):
        cells = variants[name]
        level, v, within = cells["level_a0"], cells["level_v"], cells["within"]
        print(f"  [{name}] level Q(a0): truth {level['truth_mean']:.2f} (between sd {level['between_sd']:.3f}), critic bias "
              f"{level['bias']:+.3f}, err sd {level['error_sd']:.3f} ({level['error_sd_over_between']:.2f} of between), corr "
              f"{fmt(level['corr_corrected'], '.3f')}* | V bias {v['bias']:+.3f} | within: true sd {fmt(within['true_sd'], '.4f')}, "
              f"critic sd {within['critic_sd']:.4f} (x{fmt(within['sd_ratio'], '.2f')}), corr {within['corr']:.3f} "
              f"({fmt(within['corr_corrected'], '.3f')}*, rel {within['reliability']:.2f}), pick gain "
              f"{within['critic_pick_gain']:+.4f}+-{within['critic_pick_gain_se']:.4f} "
              f"({fmt(within['critic_pick_gain_sd'], '+.2f')} sd); HL-Gauss bin width {header['hl_bin_width']:.2f}", flush=True)
        if name != primary:
            continue
        print("    gradient  space  h      n  slope true-on-critic | ascent ratio | pooled corr (corr*) |c|/|d| "
              "| cos med (ceil, half) | sign (sig, frac) | critic FD: slope, corr*, ~grad | signal frac, se rms", flush=True)
        for space, by_h in cells["gradient"].items():
            for step, g in by_h.items():
                if not g["states"]:
                    print(f"              {space:6s} {step:5s}    0", flush=True)
                    continue
                print(f"              {space:6s} {step:5s} {g['states']:4d}  {g['slope_true_on_critic']:+.3f}+-{g['slope_se']:.3f} "
                      f"| {g['ascent_ratio']:+.3f}+-{g['ascent_se']:.3f} | {g['pooled_corr']:+.3f} "
                      f"({fmt(g['pooled_corr_corrected'], '+.3f')}) {fmt(g['norm_ratio_critic_over_true'], '.2f')} | "
                      f"{g['cos_median']:+.3f} ({g['cos_ceiling_median']:.2f}, {g['split_half_cos_median']:+.2f}) | "
                      f"{g['sign_agreement']:.2f} ({fmt(g['sign_agreement_significant'], '.2f')}, {g['significant_frac']:.2f}) | "
                      f"{g['critic_fd_slope']:+.3f}, {fmt(g['critic_fd_corr_corrected'], '+.3f')}, "
                      f"{g['critic_fd_vs_autograd_cos_median']:+.2f} | {g['signal_frac']:.3f}, {g['se_rms']:.3f}", flush=True)
        print("    linearity (corr*, norm ratio): " + "; ".join(
            f"{space} {pair} {fmt(row.get('corr_corrected'), '.3f')}, {fmt(row.get('norm_ratio_small_over_large'), '.2f')}"
            for space, pairs in cells["linearity"].items() for pair, row in pairs.items()), flush=True)
        r = cells["residual"]
        print(f"    Bellman residual: mean {r['residual_mean']:+.3f} sd {r['residual_sd']:.3f} vs true err sd {r['error_sd']:.3f}; "
              f"corr(residual, -err) {r['corr_residual_neg_error']:+.3f} ({fmt(r['corr_corrected'], '+.3f')}*), slope "
              f"{r['slope_neg_error_on_residual']:+.3f}; within-state {r['within_corr']:+.3f} "
              f"({fmt(r['within_corr_corrected'], '+.3f')}*)", flush=True)
    print("  model-based one-step gradient r + gamma V'_critic(f(s, a)) vs direct critic, against the true derivative at H: "
          "slope true-on-predictor (ascent ratio) per predictor | reward-only share of |g_mb|^2 | cos(g_mb, critic) median", flush=True)
    for space, by_h in result["model_based"].items():
        for step, cell in by_h.items():
            if not cell["critic_fd"]["states"]:
                continue
            print(f"    {space:6s} {step:5s} " + "  ".join(
                f"{name}: {cell[name]['slope_true_on_critic']:+.3f}+-{cell[name]['slope_se']:.3f} "
                f"({cell[name]['ascent_ratio']:+.3f}+-{cell[name]['ascent_se']:.3f})"
                for name in ("model_soft", "model_hard", "critic_fd"))
                  + f" | {fmt(cell['reward_share'], '.3f')} | {fmt(cell['model_vs_critic_cos_median'], '+.3f')}", flush=True)
    print(f"  horizon sweep [soft, bootstrapped by the critic at t]: t | bias, err sd | within corr | slope true-on-critic "
          f"per (space, h)", flush=True)
    for t, cells in result["horizons"].items():
        level = cells["level_a0"]
        print(f"    {t:>4} | {level['bias']:+.3f}, {level['error_sd']:.3f} | {cells['within']['corr']:+.3f} | " + "  ".join(
            f"{space[0]}{step}: " + ("-" if not g["states"] else f"{g['slope_true_on_critic']:+.2f}+-{g['slope_se']:.2f}")
            for space, by_h in cells["gradient"].items() for step, g in by_h.items()), flush=True)


def run_snapshot(path, cli, device, pool):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    seed_hash = int(awr.mix32(cli.seed + 0xC21C))
    physics = awr.Physics(cli.threads, cli.seed)
    frozen = Frozen(snapshot, cli.trainer, physics, device, seed_hash)
    if cli.directions > frozen.dims or cli.rollouts < 4 or cli.rollouts % 2 or cli.fresh < 2:
        raise ValueError("--directions <= action dim, --rollouts even and >= 4, --fresh >= 2")
    per_state = 2 + cli.fresh + 2 * len(SPACES) * len(cli.steps) * cli.directions
    header = {"snapshot": str(path), "global_step": int(snapshot["global_step"]), "states": cli.states,
              "fresh": cli.fresh, "directions": cli.directions, "steps": cli.steps, "rollouts": cli.rollouts,
              "horizon": cli.horizon, "gamma": frozen.gamma, "gamma_h": frozen.gamma ** cli.horizon,
              "reward_scale": frozen.reward_scale, "temperature": frozen.temperature, "actions_per_state": per_state,
              "bootstrap_samples": cli.bootstrap_samples, "checkpoints": cli.checkpoints,
              "hl_bin_width": (snapshot["args"]["v_max"] - snapshot["args"]["v_min"]) / (snapshot["args"]["num_bins"] - 1)}
    print(json.dumps(header), flush=True)

    start = time.perf_counter()
    validation = {"normals": validate_normals(device, seed_hash), "physics": awr.validate_physics(frozen, cli.seed),
                  "returns": episode_returns(frozen, cli.validate_episodes, cli.seed)}
    returns, rate = validation["returns"], validation["physics"]["env_steps_per_s"]
    total = cli.states * per_state * cli.rollouts * cli.horizon
    print(f"validation [{time.perf_counter() - start:.0f}s]: normals KS {validation['normals']['ks']:.4f}; physics bitwise vs "
          f"gymnasium over {TIME_LIMIT} steps x {validation['physics']['copies']} copies (reward err "
          f"{validation['physics']['max_reward_error']:.1e}); returns {returns['simulator_mean']:.1f}+-{returns['simulator_sd']:.1f} "
          f"vs gymnasium {returns['gymnasium_mean']:.1f}+-{returns['gymnasium_sd']:.1f} (n {returns['episodes']} each) "
          f"z={returns['z']:+.2f}; oracle {total / 1e6:.0f}M env steps, ~{total / rate / 60:.1f} min at "
          f"{rate / 1e6:.2f}M steps/s", flush=True)
    assert abs(returns["z"]) < 4.0, f"frozen-policy return mismatch: z={returns['z']:.2f}"

    start = time.perf_counter()
    data = collect(frozen, cli.states, cli.collect_steps, cli.seed)
    obs, actions, layout = action_set(frozen, data, cli)
    critic, projected, grad = critic_readout(frozen, obs, actions, layout)
    out = branch_returns(frozen, data, actions, cli, pool)
    seconds = time.perf_counter() - start
    # Common random numbers: the duplicate of a0 must reproduce a0's returns (up to the critic's batched rounding
    # in the bootstraps), or action differences would carry noise the SEs do not describe.
    crn_error = float(np.abs(out["q"][:, :, -1] - out["q"][:, :, 0]).max())
    print(f"CRN check: max |Q(a0 duplicate) - Q(a0)| = {crn_error:.1e} (rollout sd {out['q'][0, :, 0].std(-1).mean():.3f})",
          flush=True)
    assert crn_error < 1e-3, "branches of one (state, action, rollout) diverged: common random numbers are broken"
    validation["crn_duplicate_max_error"] = crn_error
    # One-step Bellman residual r + gamma V'_critic(s') - Q_critic(s, a): the t = 1 checkpoint, averaged over m.
    residual = out["checks"][0, 0].mean(-1) - critic
    h, valid = layout["h"].double().cpu().numpy(), layout["valid"].cpu().numpy()
    tail = out["tail"].mean((1, 2))
    q_noise = (out["tail"].mean(1).var(-1, ddof=1) / cli.rollouts).mean()  # CRN-correlated across actions
    result = {"header": header, "validation": validation, "oracle_seconds": seconds,
              "bootstrap": {"tail_mean": float(tail.mean()), "tail_between_sd": float(math.sqrt(max(tail.var() - q_noise, 0.0))),
                            "limit_cut_frac": float((TIME_LIMIT - data["age"] <= cli.horizon).mean()),
                            "limit_weight_mean": float(out["limit_weight"][TIME_LIMIT - data["age"] <= cli.horizon].mean())
                            if (TIME_LIMIT - data["age"] <= cli.horizon).any() else None},
              "action_pairs": {"action_space_clamped_frac": float((h[:, 0] < np.asarray(cli.steps)[:, None]).mean()),
                               "action_space_dropped_frac": float((~valid[:, 0]).mean()),
                               "saturated_state_frac": float((np.abs(data["action"]) > 0.99).any(-1).mean())},
              "critic_grad_norm_mean": float(np.linalg.norm(grad, axis=-1).mean()),
              "variants": {name: analyze(out["q"][i], critic, projected, h, valid, residual, cli.fresh, cli.steps)
                           for i, name in enumerate(VARIANTS)},
              "model_based": model_based(
                  out["q"][0], {"model_soft": out["checks"][0, 0].mean(-1), "model_hard": out["checks"][1, 0].mean(-1),
                                "critic_fd": critic}, out["first_reward"], projected, h, valid, cli.fresh, cli.steps),
              "horizons": {str(t): analyze(out["checks"][0, j], critic, projected, h, valid, residual, cli.fresh, cli.steps)
                           for j, t in enumerate(cli.checkpoints)}}
    print_report(result)
    arrays = {"q": out["q"].astype(np.float32), "checks_soft": out["checks"][0].astype(np.float32),
              "checkpoints": np.asarray(cli.checkpoints), "critic": critic, "projected": projected, "grad": grad, "h": h,
              "valid": valid, "residual": residual, "age": data["age"], "row": data["row"], "a0": data["action"],
              "pre0": data["pre"], "directions": layout["directions"].cpu().numpy(), "tail": out["tail"].astype(np.float32)}
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER, help="the trainer file that saved them")
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--fresh", type=int, default=8, help="K fresh policy actions per state")
    parser.add_argument("--directions", type=int, default=6, help="J orthonormal difference directions")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.03, 0.1, 0.3], help="difference steps h")
    parser.add_argument("--rollouts", type=int, default=16, help="M CRN rollouts per (state, action)")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100, 200],
                        help="horizons at which Q is also bootstrapped by the critic (1 = the Bellman residual's)")
    parser.add_argument("--bootstrap-samples", type=int, default=16, help="policy actions per V' bootstrap")
    parser.add_argument("--collect-steps", type=int, default=1024, help="steps per lane the states are drawn from")
    parser.add_argument("--group-states", type=int, default=64, help="states per branch group")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--validate-episodes", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="JSON report; per-snapshot arrays go next to it (.npz)")
    cli = parser.parse_args()
    cli.steps = sorted(cli.steps)
    cli.checkpoints = sorted(set(cli.checkpoints) | {1, cli.horizon})
    if not all(1 <= t <= cli.horizon for t in cli.checkpoints):
        raise ValueError("--checkpoints must lie in [1, --horizon]")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    report = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for path in cli.snapshots:
            result, arrays = run_snapshot(path, cli, device, pool)
            report.append(result)
            if cli.out is not None:
                cli.out.parent.mkdir(parents=True, exist_ok=True)
                cli.out.write_text(json.dumps(report, indent=1))
                np.savez_compressed(cli.out.with_name(f"{cli.out.stem}_{result['header']['global_step']}.npz"), **arrays)


if __name__ == "__main__":
    main()
