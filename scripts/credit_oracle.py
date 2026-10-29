"""Credit oracle: how much does one on-policy rollout say about the ideal policy update, and where does
advantage noise come from? Ground truth from the deterministic HalfCheetah-v4 simulator (run through mlq, CUDA).

    python scripts/credit_oracle.py --control SNAPSHOT.pt [...] --reppo SNAPSHOT.pt [...]

Machinery is reused from scripts/awr_oracle_advantage.py (bitwise-validated MuJoCo branch kernel over flat
(qpos, qvel, qacc_warmstart) rows, counter-hash coupled Beta sampler) and scripts/reppo_oracle.py (counter-hash
tanh-Gaussian, REPPO soft value). Every random draw is a hash of (stream, state, sequence, step, dim), so a future
noise sequence xi = (u_1 .. u_{H-1}) is replayed exactly under any first action (common random numbers). The
coupled Beta is Marsaglia-Tsang on hashed uniforms (KS-validated), not betaincinv: equally exact CRN, on GPU.

Units. Control (sepclip PPO): r_n = clip(r_raw / sqrt(Var_ret[row 0] + 1e-8), +-10), the trainer's
VectorRewardNorm frozen at the snapshot (row 0 of 16 env rows), observations likewise (row 0). REPPO: r = 0.1 r_raw
(reward_scale) and the soft objective adds -T log pi(a_k | s_k) for k >= 1. Returns over H steps, gamma 0.99,
continuing across the 1000-step time limit (the trainer bootstraps truncations, so the continuing objective is the
one it optimizes): G_mc = sum_{t<H} gamma^t r_t (no critic) and G_boot = G_mc + gamma^H V_phi(s_H).

Per selected state s (N states from one stochastic episode, t >= --skip):
  g*(s)   = dQ/da at the policy mean action mu(s), central differences (one-sided at the bounds) at delta,
            averaged over M CRN sequences (independent of every per-sample estimator's noise), with SEs.
  gsmooth = d/dmu E_{a ~ pi_mu}[Q(s, a)]: the same FD applied to the policy's location (concentration / std fixed),
            CRN over the first-action draw too. The score estimator is unbiased for gsmooth, not for g*.
  grid    = K on-policy actions a_k x M_g sequences: two-way ANOVA of G into action effect, common future luck
            (same sequence hurts every action alike) and action x luck interaction; V_true = grid mean.
  samples = realized (a_k, xi_k) (grid diagonal) + E extra independent rollouts: MC advantage G - b_phi(s),
            the trainer-style lambda advantage along the realized trajectory, and estimators of g*:
            (i) score A * d/dmu log pi(a | s) (location score, concentration / std held fixed), with A from MC with the
                critic baseline, MC with the true baseline, GAE, and the oracle A_true(s, a_k) (luck removed),
            (ii) pathwise hindsight dG/da at a_k holding the realized xi_k fixed,
            (iii) one-step d/da [r(s, a) + gamma V_phi(s'(s, a))] with the true simulator,
            (iv, REPPO) the learned critic's autograd dQ/da.
  Cosines with g* in plain action space and in the policy's natural metric (inner product with F^-1, F the
  location Fisher), and the number of samples an average needs to reach cosine 0.9, from per-state signal
  and U-statistic mean energy (noise-corrected for g*'s own SE).
"""

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awr_oracle_advantage as base  # noqa: E402  (puts the repo root on sys.path)
import reppo_latentroll_oracle as latentroll  # noqa: E402
import reppo_oracle as reppo  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ENV_ID = base.ENV_ID
# Hash streams, disjoint from each other by their second word.
COLLECT, FIRST, BRANCH, VALUE, EPISODE = 21, 22, 23, 24, 25
GRID_SEQ, EXTRA_SEQ, ONESTEP_SEQ, SMOOTH_SEQ = 100_000, 200_000, 300_000, 400_000  # g* uses 0..M-1
SMOOTH_FIRST = 50_000  # FIRST-stream indices of the smoothed target's first-action draws


# ---------------------------------------------------------------- policies

class ControlPolicy:
    """The sepclip PPO control: Beta actor + V critic, frozen VectorObsNorm / VectorRewardNorm (row `row`)."""

    kind = "control"

    def __init__(self, snapshot, physics, device, cli):
        trainer = importlib.import_module(snapshot["trainer_module"])
        env = gym.make(ENV_ID)
        spaces = SimpleNamespace(single_action_space=env.action_space,
                                 single_observation_space=env.observation_space)
        env.close()
        self.agent = trainer.Agent(spaces).to(device)
        self.agent.load_state_dict(snapshot["model"])
        self.agent.eval().requires_grad_(False)
        self.physics, self.device, self.candidates = physics, device, cli.candidates
        args = snapshot["args"]
        self.gamma, self.lam, self.temperature = float(args["gamma"]), float(args["gae_lambda"]), 0.0
        as64 = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)  # noqa: E731
        row = cli.norm_row
        self.obs_mean = as64(snapshot["obs_norm"]["means"][row])
        self.obs_scale = torch.sqrt(as64(snapshot["obs_norm"]["variances"][row]) + base.NORM_EPSILON)
        self.reward_scale = math.sqrt(float(snapshot["rew_norm"]["variances"][row]) + base.NORM_EPSILON)
        self.low, self.span = self.agent.action_low, self.agent.action_high - self.agent.action_low
        self.dims = torch.arange(self.agent.action_dim, device=device)
        self.action_dim = self.agent.action_dim
        self.units = (f"r_raw / {self.reward_scale:.3f} (VectorRewardNorm row {row}: sqrt(Var_ret + 1e-8)), "
                      f"clipped to +-10")

    def observe(self, state):
        obs = state[:, 1:self.physics.nq + self.physics.nv]
        return ((obs - self.obs_mean) / self.obs_scale).clamp(-base.NORM_CLIP, base.NORM_CLIP).float()

    def raw_reward(self, x_after, x_before, action):
        forward = base.FORWARD_REWARD_WEIGHT * ((x_after - x_before) / self.physics.dt)
        return forward - base.CTRL_COST_WEIGHT * action.double().square().sum(-1)

    def step_reward(self, raw):
        return (raw / self.reward_scale).clamp(-base.NORM_CLIP, base.NORM_CLIP)

    def _concentrations(self, obs):
        alpha, beta, _ = self.agent.get_policy_and_value(obs)
        return alpha, beta

    def _native(self, alpha, beta, key):
        return base.coupled_beta(alpha, beta, base.fold(key[..., None], self.dims), self.candidates)

    def act(self, obs, key):
        alpha, beta = self._concentrations(obs)
        return (self.low + self.span * self._native(alpha, beta, key)).contiguous(), None

    def mean_action(self, obs):
        alpha, beta = self._concentrations(obs)
        return self.low + self.span * alpha / (alpha + beta)

    def sample_with_score(self, obs, key):
        """Action, location score d/dmu log pi(a) in env units (kappa = alpha + beta fixed) and its Fisher."""
        alpha, beta = (x.double() for x in self._concentrations(obs))
        native = self._native(alpha.float(), beta.float(), key)
        action = (self.low + self.span * native).contiguous()
        x = native.double()
        kappa, span = alpha + beta, self.span.double()
        score = kappa * ((x.log() - torch.log1p(-x)) - (torch.digamma(alpha) - torch.digamma(beta))) / span
        fisher = kappa.square() * (torch.polygamma(1, alpha) + torch.polygamma(1, beta)) / span.square()
        return action, score, fisher

    def sample_shifted(self, obs, key, location):
        """The policy's action draw (same key) after moving its mean to `location` (env units), kappa fixed."""
        alpha, beta = self._concentrations(obs)
        kappa = alpha + beta
        m = ((location - self.low) / self.span).clamp(1e-4, 1.0 - 1e-4)
        native = self._native(kappa * m, kappa * (1.0 - m), key)
        return (self.low + self.span * native).contiguous()

    def value(self, obs, key):
        return self.agent.get_value(obs).flatten().double()

    def baseline(self, obs, key, samples=None):
        return self.value(obs, key)  # PPO's advantage baseline is V_phi itself

    critic_action_grad = None


class ReppoPolicy:
    """REPPO: tanh-Gaussian actor + categorical Q critic (reppo_oracle.Oracle), soft objective at temperature T."""

    kind = "reppo"

    def __init__(self, snapshot, physics, device, cli):
        trainer, self.trainer_path = latentroll.load_trainer(snapshot)
        self.trainer = trainer
        ns = SimpleNamespace(value_samples=cli.value_samples, seed=cli.seed)
        self.oracle = reppo.Oracle(trainer, snapshot, physics, ns, device)
        self.physics, self.device = physics, device
        self.gamma, self.lam = self.oracle.gamma, self.oracle.lam
        self.temperature, self.reward_scale = self.oracle.temperature, self.oracle.reward_scale
        self.action_dim = self.oracle.action_dim
        self.row = torch.full((1,), cli.norm_row, device=device, dtype=torch.long)
        self.units = f"{self.reward_scale} x r_raw (reward_scale); soft returns add -T log pi, T={self.temperature:.4g}"

    def observe(self, state):
        return self.oracle.observe(state, self.row.expand(state.shape[0]))

    raw_reward = ControlPolicy.raw_reward

    def step_reward(self, raw):
        return self.reward_scale * raw

    def act(self, obs, key):
        action, logprob = self.oracle.sample(obs, key)
        return action.contiguous(), logprob

    def mean_action(self, obs):
        mean, _ = self.oracle.moments(obs)
        return torch.tanh(mean)

    def sample_with_score(self, obs, key):
        """Location score w.r.t. mu = tanh(m) (std fixed): (z / sigma) / (1 - mu^2), Fisher 1 / (sigma (1 - mu^2))^2."""
        mean, std = self.oracle.moments(obs)
        z = reppo.normals(key, self.action_dim)
        action = torch.tanh(mean + std * z).contiguous()
        jac = (1.0 - torch.tanh(mean).double().square()).clamp_min(1e-12)
        score = (z.double() / std.double()) / jac
        fisher = 1.0 / (std.double() * jac).square()
        return action, score, fisher

    def sample_shifted(self, obs, key, location):
        """tanh(atanh(location) + sigma z) with the same z: the draw after moving the mean action, std fixed."""
        _, std = self.oracle.moments(obs)
        z = reppo.normals(key, self.action_dim)
        pre = torch.atanh(location.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        return torch.tanh(pre + std * z).contiguous()

    def value(self, obs, key):
        """Soft V(s) = E_a[Q_phi(s, a) - T log pi(a | s)], CRN draws from key."""
        return self.oracle.soft_value(obs, key)

    def baseline(self, obs, key, samples=None):
        """E_{a ~ pi} Q_phi(s, a): the baseline an advantage of the first action is measured against."""
        samples = samples or self.oracle.value_samples
        mean, std = self.oracle.moments(obs)
        noise = reppo.normals(base.fold(key[:, None], torch.arange(samples, device=self.device)), self.action_dim)
        action, _ = reppo.squash(mean[:, None], std[:, None], noise)
        return self.oracle.critic_q(obs, action).mean(1).double()

    def critic_action_grad(self, obs, actions):
        return self.oracle.critic_grad(obs, actions).double()


# ---------------------------------------------------------------- simulation

@torch.no_grad()
def run_branches(policy, physics, states, sids, seqs, first, record, horizon, chunk, seed_hash, b_phi):
    """Roll every branch (start row, first action, then the policy on sequence `seq`) for `horizon` steps.

    Returns mc = sum_{t<H} gamma^t r_t, bonus = sum_{1<=k<H} gamma^k (-T log pi(a_k | s_k)), tail = V(s_H), and for
    `record` branches the per-step rewards, soft bonuses and V along the trajectory (V(s_0) = baseline)."""
    device, gamma = policy.device, policy.gamma
    count = len(states)
    mc = torch.zeros(count, dtype=torch.float64, device=device)
    bonus = torch.zeros_like(mc)
    tail = torch.zeros_like(mc)
    rec_index = torch.nonzero(record).flatten()
    rec_pos = torch.full((count,), -1, dtype=torch.long, device=device)
    rec_pos[rec_index] = torch.arange(len(rec_index), device=device)
    rec_rewards = torch.zeros((len(rec_index), horizon), dtype=torch.float64, device=device)
    rec_bonus = torch.zeros_like(rec_rewards)
    rec_values = torch.zeros((len(rec_index), horizon + 1), dtype=torch.float64, device=device)
    begin = time.perf_counter()
    # Chunks never split a state's branches, so every CRN comparison sees one batch shape.
    sid_np = sids.cpu().numpy()
    edges, last = [0], 0
    for cut in [int(c) for c in np.flatnonzero(np.diff(sid_np)) + 1] + [count]:
        if cut - edges[-1] > chunk and last > edges[-1]:
            edges.append(last)  # close the chunk at the previous state boundary
        last = cut
    edges.append(count)
    for lo, hi in zip(edges[:-1], edges[1:]):
        state = np.ascontiguousarray(states[lo:hi])
        sid, seq, action = sids[lo:hi], seqs[lo:hi], first[lo:hi].contiguous()
        rec = record[lo:hi]
        rpos = rec_pos[lo:hi][rec]
        if rec.any():
            rec_values[rpos, 0] = b_phi[sid[rec]]  # the same baseline the MC advantage uses
        for t in range(horizon):
            x_before = torch.from_numpy(state[:, 0].copy()).to(device)
            physics.step(state, action.cpu().numpy())
            current = torch.from_numpy(state).to(device)
            r = policy.step_reward(policy.raw_reward(current[:, 0], x_before, action))
            mc[lo:hi] += gamma ** t * r
            obs = policy.observe(current)
            if rec.any():
                rec_rewards[rpos, t] = r[rec]
                rec_values[rpos, t + 1] = policy.value(obs[rec], base.stream_key(seed_hash, VALUE, sid[rec], seq[rec], t + 1))
            if t < horizon - 1:
                action, logprob = policy.act(obs, base.stream_key(seed_hash, BRANCH, sid, seq, t + 1))
                if policy.temperature > 0.0:
                    soft = -policy.temperature * logprob.double()
                    bonus[lo:hi] += gamma ** (t + 1) * soft
                    if rec.any():
                        rec_bonus[rpos, t + 1] = soft[rec]
        tail[lo:hi] = policy.value(obs, base.stream_key(seed_hash, VALUE, sid, seq, horizon))
        elapsed = time.perf_counter() - begin
        print(f"  branches {hi}/{count}: {elapsed:.0f}s, {hi * horizon / elapsed / 1e6:.2f}M env steps/s", flush=True)
    return {"mc": mc, "bonus": bonus, "tail": tail, "rec_index": rec_index, "rec_rewards": rec_rewards,
            "rec_bonus": rec_bonus, "rec_values": rec_values}


@torch.no_grad()
def collect(policy, physics, steps, seed_hash):
    """One stochastic episode from gymnasium's seeded reset: kernel rows before every step, actions, raw rewards."""
    state = physics.reset(1)
    rows = np.empty((steps + 1, physics.width))
    actions = np.empty((steps, policy.action_dim), dtype=np.float32)
    rewards = np.empty(steps)
    for t in range(steps):
        rows[t] = state[0]
        current = torch.from_numpy(state).to(policy.device)
        action, _ = policy.act(policy.observe(current), base.stream_key(seed_hash, COLLECT, torch.zeros(1, dtype=torch.long,
                                                                                                       device=policy.device), t))
        actions[t] = action.cpu().numpy()
        physics.step(state, actions[t:t + 1])
        rewards[t] = float(policy.raw_reward(torch.from_numpy(state[:, 0]), current[:, 0].cpu(),
                                             torch.from_numpy(actions[t:t + 1]))[0])
    rows[steps] = state[0]
    return rows, actions, rewards


def validate_restore(policy, physics, rows, actions, rewards, starts, length):
    """Restore saved rows into (a) the kernel and (b) gymnasium's own env, replay the recorded actions, and require
    bit-identical rows/observations and identical rewards against the original trajectory. gymnasium computes the
    control cost in float32 (np.square of the float32 action), the oracle in float64: that gap (~1e-7) is reported."""
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=0)
    nq, nv = physics.nq, physics.nv
    worst, worst_gym = 0.0, 0.0
    for t0 in starts:
        state = rows[t0:t0 + 1].copy()
        env.data.qpos[:], env.data.qvel[:], env.data.qacc_warmstart[:] = rows[t0, :nq], rows[t0, nq:nq + nv], rows[t0, nq + nv:]
        for k in range(length):
            t = t0 + k
            physics.step(state, actions[t:t + 1])
            obs, reward, *_ = env.step(actions[t])
            assert np.array_equal(state[0], rows[t + 1]), f"kernel restore diverged at t={t}"
            assert np.array_equal(obs, rows[t + 1, 1:nq + nv]), f"gymnasium restore diverged at t={t}"
            mine = float(policy.raw_reward(torch.from_numpy(state[:, 0]), torch.from_numpy(rows[t:t + 1, 0]),
                                           torch.from_numpy(actions[t:t + 1]))[0])
            worst = max(worst, abs(mine - rewards[t]))
            worst_gym = max(worst_gym, abs(reward - rewards[t]))
    env.close()
    assert worst < 1e-9 and worst_gym < 1e-6, f"restored reward mismatch {worst} / gymnasium {worst_gym}"
    return {"starts": [int(s) for s in starts], "steps_each": length, "kernel_rows_bitwise": True,
            "gymnasium_obs_bitwise": True, "max_reward_abs_error_kernel": worst,
            "max_reward_abs_error_vs_gymnasium_float32_ctrl_cost": worst_gym}


@torch.no_grad()
def episode_returns(policy, physics, episodes, seed_hash):
    state = physics.reset(episodes)
    lanes = torch.arange(episodes, device=policy.device)
    total = torch.zeros(episodes, dtype=torch.float64, device=policy.device)
    for t in range(base.TIME_LIMIT):
        before = torch.from_numpy(state[:, 0].copy()).to(policy.device)
        action, _ = policy.act(policy.observe(torch.from_numpy(state).to(policy.device)),
                               base.stream_key(seed_hash, EPISODE, lanes, t))
        physics.step(state, action.cpu().numpy())
        total += policy.raw_reward(torch.from_numpy(state[:, 0]).to(policy.device), before, action)
    return total.cpu().numpy()


# ---------------------------------------------------------------- statistics

def probes(center, delta):
    """(n, D, 2, D) plus/minus probes clipped to [-1, 1], and (n, D) spans a+ - a-."""
    dims = center.shape[-1]
    eye = torch.eye(dims, device=center.device, dtype=center.dtype) * delta
    plus = (center[:, None] + eye).clamp(-1.0, 1.0)
    minus = (center[:, None] - eye).clamp(-1.0, 1.0)
    span = (plus - minus).diagonal(dim1=1, dim2=2)
    return torch.stack((plus, minus), 2), span


def anova(grid):
    """Two-way random-effects ANOVA of G[k, m]: action, common luck (sequence), interaction; unbiased."""
    k, m = grid.shape
    gk, gm, g = grid.mean(1), grid.mean(0), grid.mean()
    msa = m * ((gk - g) ** 2).sum() / (k - 1)
    msb = k * ((gm - g) ** 2).sum() / (m - 1)
    mse = ((grid - gk[:, None] - gm[None] + g) ** 2).sum() / ((k - 1) * (m - 1))
    action, common = (msa - mse) / m, (msb - mse) / k
    v_se2 = max(action, 0.0) / k + max(common, 0.0) / m + mse / (k * m)
    return {"action": float(action), "common_luck": float(common), "interaction": float(mse),
            "V_true": float(g), "V_true_se2": float(v_se2)}


def estimator_stats(samples, g, se, w):
    """Cosine of each sample (or the single estimate) with g* under <u, v> = sum w u v, and the averaged
    estimator's growth: cos_N = par / sqrt(m2 + (q - m2) / N), par = E<g_hat, g*>/|g*_true|, q = E|g_hat|^2,
    m2 = |E g_hat|^2 (U-statistic), |g*_true|^2 = |g*|^2 - sum w SE^2."""
    samples = np.atleast_2d(samples)
    gg = float((w * g * g).sum())
    gtrue2 = gg - float((w * se * se).sum())
    dots = samples @ (w * g)
    norms2 = (w * samples * samples).sum(1)
    out = {"n": len(samples), "cos_raw": float(np.mean(dots / np.sqrt(np.maximum(norms2 * gg, 1e-300))))}
    # Energy-weighted pooling terms (estimator scaled to unit rms; all three unbiased, usable when gtrue2 <= 0):
    # A = E<g_hat, g*_est>, B = |E g_hat|^2 (U-statistic; 1 for a deterministic estimate), C = |g*_true|^2.
    q = float(norms2.mean())
    if len(samples) >= 2:
        total = samples.sum(0)
        b = (float((w * total * total).sum()) - float(norms2.sum())) / (len(samples) * (len(samples) - 1)) / q
    else:
        b = 1.0
    out["ew"] = (float(dots.mean()) / math.sqrt(q), b, gtrue2)
    if gtrue2 <= 0:
        return out
    gt = math.sqrt(gtrue2)
    out["cos_corr"] = float(np.mean(dots / (np.sqrt(np.maximum(norms2, 1e-300)) * gt)))
    if len(samples) >= 2:
        n, q, total = len(samples), float(norms2.mean()), samples.sum(0)
        m2 = (float((w * total * total).sum()) - float(norms2.sum())) / (n * (n - 1))
        par = float(dots.mean()) / gt
        out.update(q=q, m2=m2, par=par, par_n=par / math.sqrt(q), m2_n=m2 / q,
                   cos_inf=par / math.sqrt(m2) if m2 > 0 else float("nan"), n09=samples_to(par / math.sqrt(q), m2 / q))
    return out


def samples_to(par_n, m2_n, target=0.9):
    """N such that par_n / sqrt(m2_n + (1 - m2_n) / N) = target (normalized so q = 1); inf if unreachable."""
    need = (par_n / target) ** 2 - m2_n
    if par_n <= 0 or need <= 0:
        return float("inf")
    return max((1.0 - m2_n) / need, 1.0)


def energy_weighted(terms):
    """cos1 = sum A / sqrt(S sum C), cos_inf = sum A / sqrt(sum B sum C), N_0.9 from
    cos_N = sum A / sqrt(sum C (sum B + (S - sum B) / N)): states weighted by their true-gradient energy."""
    a, b, c = (float(x) for x in np.asarray(terms).sum(0))
    count = len(terms)
    if c <= 0:
        return float("nan"), float("nan"), float("nan")
    cos1 = a / math.sqrt(count * c)
    cos_inf = a / math.sqrt(b * c) if b > 0 else float("nan")
    need = a * a / (0.81 * c) - b
    n09 = float("inf") if a <= 0 or need <= 0 else max((count - b) / need, 1.0)
    return cos1, cos_inf, n09


def pool(stats_list, boots=2000, seed=0):
    """Typical-state summary over states: means of per-state normalized signal/energy, then N_0.9; and the
    energy-weighted pooled cosines over all states with a state-bootstrap 90% interval."""
    terms = np.array([s["ew"] for s in stats_list])
    cos1, cos_inf, n09 = energy_weighted(terms)
    rng = np.random.default_rng(seed)
    draws = np.array([energy_weighted(terms[rng.integers(0, len(terms), len(terms))]) for _ in range(boots)])
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(draws[:, :2], [5, 95], axis=0)
        n_lo, n_hi = np.nanpercentile(np.where(np.isfinite(draws[:, 2]), draws[:, 2], 1e12), [5, 95])
    ew = {"cos1": cos1, "cos1_ci90": [float(lo[0]), float(hi[0])], "cos_inf": cos_inf,
          "cos_inf_ci90": [float(lo[1]), float(hi[1])], "n09": n09, "n09_ci90": [float(n_lo), float(n_hi)]}
    cells = [s for s in stats_list if "par_n" in s]
    singles = [s for s in stats_list if "cos_corr" in s]
    out = {"energy_weighted": ew, "states": len(stats_list), "states_scored": len(singles),
           "cos_raw_mean": float(np.mean([s["cos_raw"] for s in stats_list])) if stats_list else float("nan")}
    if singles:
        out["cos_corr_mean"] = float(np.mean([s["cos_corr"] for s in singles]))
    if cells:
        par_n, m2_n = float(np.mean([s["par_n"] for s in cells])), float(np.mean([s["m2_n"] for s in cells]))
        n09 = np.array([s["n09"] for s in cells])
        out.update(par_n=par_n, m2_n=m2_n, cos_inf=par_n / math.sqrt(m2_n) if m2_n > 0 else float("nan"),
                   n09_pooled=samples_to(par_n, m2_n), n09_median_state=float(np.median(n09)),
                   frac_states_reachable=float(np.isfinite(n09).mean()),
                   cos_at={str(n): par_n / math.sqrt(max(m2_n + (1 - m2_n) / n, 1e-12)) for n in (1, 16, 128, 1024)})
    return out


# ---------------------------------------------------------------- one snapshot

def analyze(raw, meta, cli, label):
    """Every statistic, from a snapshot's raw branch returns (also used by --rescore)."""
    variants, layout, a_gae, onestep, critic_grad = (raw[k] for k in ("variants", "layout", "a_gae", "onestep", "critic_grad"))
    score_np, fisher_np, b_phi_np, mu_np = raw["score"], raw["fisher"], raw["b_phi"], raw["mu"]
    rec_pos = {int(b): j for j, b in enumerate(raw["rec_index"])}
    picks = np.asarray(meta["state_times"])
    n, D = len(layout), score_np.shape[-1]
    K, Mg, E, M = meta["K"], meta["M_grid"], meta["E_extra"], meta["M_gstar"]
    results = {}
    for vname, G in variants.items():
        per_state = []
        est = {}
        batch = {"a_true": [], "a_mc": [], "a_gae": [], "luck": [], "b_err": [], "v_se2": []}
        sens = []
        for i, cell in enumerate(layout):
            row = {"t": int(picks[i])}
            gstars = {}
            halfcos = lambda fd: float(fd[::2].mean(0) @ fd[1::2].mean(0)  # noqa: E731
                                       / (np.linalg.norm(fd[::2].mean(0)) * np.linalg.norm(fd[1::2].mean(0))))
            for delta, (idx, span) in cell["gstar"].items():
                fd = (G[idx[..., 0]] - G[idx[..., 1]]) / span  # (M, D)
                gstars[delta] = (fd.mean(0), fd.std(0, ddof=1) / math.sqrt(M))
                if delta == cli.deltas[0]:
                    row["gstar_split_half_cos"] = halfcos(fd)
            g, se = gstars[cli.deltas[0]]
            idx, span = cell["smooth"]
            fd = (G[idx[..., 0]] - G[idx[..., 1]]) / span
            gsm, gsm_se = fd.mean(0), fd.std(0, ddof=1) / math.sqrt(M)
            row["gsmooth_split_half_cos"] = halfcos(fd)
            row.update(gsmooth=gsm.tolist(), gsmooth_rel_se=float(np.linalg.norm(gsm_se) / np.linalg.norm(gsm)),
                       cos_gstar_vs_gsmooth=float(gsm @ g / (np.linalg.norm(gsm) * np.linalg.norm(g))))
            targets = {f"g*d{d}": gstars[d] for d in cli.deltas}
            targets["gsmooth"] = (gsm, gsm_se)
            row.update(gstar=g.tolist(), gstar_se=se.tolist(), gstar_rel_se=float(np.linalg.norm(se) / np.linalg.norm(g)))
            for delta, (gd, sd) in gstars.items():
                if delta != cli.deltas[0]:
                    c = float(gd @ g / (np.linalg.norm(gd) * np.linalg.norm(g)))
                    row[f"cos_gstar_{cli.deltas[0]}_vs_{delta}"] = c
                    sens.append((delta, c, float(np.linalg.norm(gd) / np.linalg.norm(g))))
            grid = G[cell["grid"]]  # (K, Mg)
            av = anova(grid)
            row.update(av)
            row["b_phi"], row["b_err"] = float(b_phi_np[i]), float(b_phi_np[i] - av["V_true"])
            # Realized samples: diagonal (a_k, xi_k), A_true from the other sequences (independent of xi_k).
            diag = grid[np.arange(K), np.arange(K)]
            mask = ~np.eye(K, Mg, dtype=bool)
            q_loo = np.array([grid[k, mask[k]].mean() for k in range(K)])
            v_loo = np.array([grid[np.arange(K) != k][:, mask[k]].mean() for k in range(K)])
            a_true = q_loo - v_loo
            a_mc = diag - b_phi_np[i]
            gae_diag = a_gae[[rec_pos[int(b)] for b in cell["grid"][np.arange(K), np.arange(K)]]]
            batch["a_true"].append(a_true); batch["a_mc"].append(a_mc); batch["a_gae"].append(gae_diag)
            batch["luck"].append(diag - q_loo); batch["b_err"].append(row["b_err"]); batch["v_se2"].append(av["V_true_se2"])
            extra_g = G[cell["extra"]]
            gae_extra = a_gae[[rec_pos[int(b)] for b in cell["extra"]]]
            # (i) score estimators over realized samples (diagonal + extras)
            sc = score_np[i]  # (K + E, D)
            ret = np.concatenate([diag, extra_g])
            true_base = np.concatenate([v_loo, np.full(E, av["V_true"])])
            samples = {
                "score_mc_critic_baseline": (ret - b_phi_np[i])[:, None] * sc,
                "score_mc_true_baseline": (ret - true_base)[:, None] * sc,
                "score_gae": np.concatenate([gae_diag, gae_extra])[:, None] * sc,
                "score_oracle_A": a_true[:, None] * sc[:K],
            }
            for delta, (idx, span) in cell["path"].items():  # (ii) hindsight pathwise on realized xi_k
                samples[f"pathwise_hindsight_d{delta}"] = (G[idx[..., 0]] - G[idx[..., 1]]) / span
            for delta, grads in onestep.items():  # (iii)
                samples[f"onestep_critic_at_ak_d{delta}"] = grads[i, 1:]
                samples[f"onestep_critic_at_mu_d{delta}"] = grads[i, :1]
            if critic_grad is not None:  # (iv)
                samples["critic_dQda_at_ak"] = critic_grad[i, 1:]
                samples["critic_dQda_at_mu"] = critic_grad[i, :1]
            for metric, w in (("plain", np.ones(D)), ("natural", 1.0 / fisher_np[i])):
                for target, (gt, st_) in targets.items():
                    for name, s in samples.items():
                        est.setdefault(f"{name}|{metric}|{target}", []).append(estimator_stats(s, gt, st_, w))
            row["fisher_diag"] = fisher_np[i].tolist()
            row["mu"] = mu_np[i].tolist()
            per_state.append(row)

        # ---- pooled variance decomposition of the realized MC advantage
        mean = lambda key: float(np.mean([r[key] for r in per_state]))  # noqa: E731
        action, common, inter = mean("action"), mean("common_luck"), mean("interaction")
        b_err = np.array(batch["b_err"])
        b2 = float(np.mean(b_err ** 2) - np.mean(batch["v_se2"]))
        b_var = float(np.var(b_err, ddof=1) - np.mean(batch["v_se2"]))
        total = action + common + inter + b2
        a_true, a_mc, a_gae_d = (np.concatenate(batch[k]) for k in ("a_true", "a_mc", "a_gae"))
        a_true_noise = float(np.mean([r["interaction"] / (Mg - 1) for r in per_state]))
        corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])  # noqa: E731

        def within(x):
            return np.concatenate([v - v.mean() for v in np.split(x, n)])

        decomposition = {
            "per_sample_variance_terms": {"action": action, "luck_common": common, "luck_interaction": inter,
                                          "critic_baseline_sq": b2, "critic_baseline_var_across_states": b_var},
            "shares_of_E[A_mc^2]": {"action": action / total, "luck_common": common / total,
                                    "luck_interaction": inter / total, "critic_baseline": b2 / total},
            "E[A_mc^2]_observed": float(np.mean(a_mc ** 2)), "E[A_mc^2]_sum_of_terms": total,
            "action_share_within_state_median": float(np.median([r["action"] / (r["action"] + r["common_luck"] + r["interaction"])
                                                                 for r in per_state])),
            "common_luck_share_within_state_median": float(np.median(
                [r["common_luck"] / (r["action"] + r["common_luck"] + r["interaction"]) for r in per_state])),
            "critic_baseline_share_median_state": float(np.median(
                [max(r["b_err"] ** 2 - r["V_true_se2"], 0.0) / (r["action"] + r["common_luck"] + r["interaction"]
                                                               + max(r["b_err"] ** 2 - r["V_true_se2"], 0.0)) for r in per_state])),
            "critic_baseline_sq_top3_state_fraction": float(np.sort(b_err ** 2)[-3:].sum() / max((b_err ** 2).sum(), 1e-300)),
            "within_state_var_top3_state_fraction": float(np.sort([r["action"] + r["common_luck"] + r["interaction"]
                                                                   for r in per_state])[-3:].sum()
                                                          / (n * (action + common + inter))),
            "critic_V_error": {"mean": float(b_err.mean()), "rms_corrected": math.sqrt(max(b2, 0.0)),
                               "var_across_states_over_mean_within_state_var_A": b_var / (action + common + inter),
                               "corr_bphi_Vtrue_across_states": corr(b_phi_np, b_phi_np - b_err)},
            "a_true_estimation_noise_var": a_true_noise,
            "advantage_fidelity": {
                "corr_mc_vs_true": corr(a_mc, a_true), "corr_gae_vs_true": corr(a_gae_d, a_true),
                "corr_within_state_mc_vs_true": corr(within(a_mc), within(a_true)),
                "corr_within_state_gae_vs_true": corr(within(a_gae_d), within(a_true)),
                "E[(A_mc-A_true)^2]/E[A_mc^2]": float(np.mean((a_mc - a_true) ** 2) / np.mean(a_mc ** 2)),
                "E[(A_gae-A_true)^2]/E[A_gae^2]": float(np.mean((a_gae_d - a_true) ** 2) / np.mean(a_gae_d ** 2)),
                "E[A_gae^2]": float(np.mean(a_gae_d ** 2)), "E[A_true^2]": float(np.mean(a_true ** 2)),
                "gae_mean_offset": float(np.mean(a_gae_d - a_true)),
            },
        }
        gstar_quality = {
            "rel_se_median": float(np.median([r["gstar_rel_se"] for r in per_state])),
            "frac_states_snr_gt_1": float(np.mean([r["gstar_rel_se"] < 1.0 for r in per_state])),
            "gsmooth_rel_se_median": float(np.median([r["gsmooth_rel_se"] for r in per_state])),
            "gstar_split_half_cos_median": float(np.median([r["gstar_split_half_cos"] for r in per_state])),
            "gsmooth_split_half_cos_median": float(np.median([r["gsmooth_split_half_cos"] for r in per_state])),
            "cos_gstar_vs_gsmooth_median": float(np.median([r["cos_gstar_vs_gsmooth"] for r in per_state])),
            "delta_sensitivity": {str(d): {"cos_median": float(np.median([c for dd, c, _ in sens if dd == d])),
                                           "norm_ratio_median": float(np.median([x for dd, _, x in sens if dd == d])),
                                           "states": sum(dd == d for dd, _, _ in sens)}
                                  for d in sorted({dd for dd, _, _ in sens})},
        }
        results[vname] = {"decomposition": decomposition, "gstar": gstar_quality,
                          "estimators": {k: pool(v) for k, v in est.items()}, "per_state": per_state}
        print(f"[{label} {vname}] shares {json.dumps({k: round(v, 3) for k, v in decomposition['shares_of_E[A_mc^2]'].items()})}"
              f" | g* rel SE {gstar_quality['rel_se_median']:.2f}", flush=True)
        for k, v in results[vname]["estimators"].items():
            if f"|g*d{cli.deltas[0]}" in k or k.endswith("|gsmooth"):
                ew = v["energy_weighted"]
                print(f"    {k:62s} raw {v['cos_raw_mean']:+.3f} | ew cos1 {ew['cos1']:+.3f} [{ew['cos1_ci90'][0]:+.2f},"
                      f"{ew['cos1_ci90'][1]:+.2f}] cos_inf {ew['cos_inf']:+.3f} N0.9 {ew['n09']:.3g}", flush=True)

    return results


def run_snapshot(kind, path, cli, device):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    assert snapshot["args"]["env_id"] == ENV_ID
    torch._dynamo.reset()
    physics = base.Physics(cli.threads, cli.seed)
    policy = (ControlPolicy if kind == "control" else ReppoPolicy)(snapshot, physics, device, cli)
    seed_hash = int(base.mix32(cli.seed + 0xC4ED17))
    step = int(snapshot["global_step"])
    label = f"{kind}_{step}"
    print(f"=== {label}: {path}\n    units: {policy.units}", flush=True)
    start = time.perf_counter()

    # Validation: sampler marginals, kernel vs gymnasium (bitwise), restore + replay, frozen-policy returns.
    validation = {"physics": base.validate_physics(policy, cli.seed)}
    if kind == "control":
        validation["sampler"] = base.validate_sampler(device, seed_hash, cli.candidates)
    else:
        validation["sampler"] = reppo.validate_sampler(policy.trainer, device, seed_hash)
    rows, actions, rewards = collect(policy, physics, cli.episode_steps, seed_hash)
    picks = np.unique(np.round(np.linspace(cli.skip, cli.episode_steps - 1, cli.states)).astype(int))
    validation["restore"] = validate_restore(policy, physics, rows, actions, rewards,
                                           picks[picks <= cli.episode_steps - 100][:: max(1, len(picks) // 8)][:8], 100)
    returns = episode_returns(policy, physics, cli.validate_episodes, seed_hash)
    validation["episode_return_raw"] = {"mean": float(returns.mean()), "sd": float(returns.std()), "n": len(returns),
                                        "collection_episode": float(rewards.sum())}
    if kind == "control":
        norm = snapshot["obs_norm"]
        validation["norm_rows"] = {
            "obs_mean_max_dev_in_sd": float(np.abs(norm["means"] - norm["means"][cli.norm_row]).max()
                                            / np.sqrt(norm["variances"][cli.norm_row]).min()),
            "reward_var_range": [float(snapshot["rew_norm"]["variances"].min()), float(snapshot["rew_norm"]["variances"].max())]}
    print(f"validation ok [{time.perf_counter() - start:.0f}s]: {json.dumps(validation['restore'])}; "
          f"episode return {returns.mean():.0f}+-{returns.std():.0f}; kernel {validation['physics']['env_steps_per_s'] / 1e6:.2f}M/s",
          flush=True)

    states = rows[picks]
    n, D = len(states), policy.action_dim
    K, Mg, E, M, H = cli.samples, cli.grid_rollouts, cli.extra, cli.rollouts, cli.horizon
    assert Mg >= K, "the realized sample k uses grid sequence k"
    sids_all = torch.arange(n, device=device)
    with torch.no_grad():
        obs0 = policy.observe(torch.from_numpy(states).to(device))
        mu = policy.mean_action(obs0).float()
        fresh_keys = base.stream_key(seed_hash, FIRST, sids_all[:, None], torch.arange(K + E, device=device))
        a_fresh, score, fisher = policy.sample_with_score(obs0[:, None].expand(-1, K + E, -1), fresh_keys)
        fisher = fisher[:, 0]  # depends on s only
        b_phi = policy.baseline(obs0, base.stream_key(seed_hash, VALUE, sids_all, -1), samples=cli.baseline_samples)
        v_phi = policy.value(obs0, base.stream_key(seed_hash, VALUE, sids_all, -1))

    # ---- branch layout
    parts, layout = [], []
    count = 0

    def add(state_i, seqs, acts, record=False):
        nonlocal count
        acts = acts.reshape(-1, D)
        seqs = torch.as_tensor(seqs, device=device).reshape(-1).expand(len(acts)) if np.ndim(seqs) == 0 else \
            torch.as_tensor(seqs, device=device).reshape(-1)
        parts.append((state_i, seqs, acts, record))
        idx = np.arange(count, count + len(acts))
        count += len(acts)
        return idx

    gstar_deltas = {i: list(cli.deltas) + (list(cli.extra_deltas) if i in set(range(0, n, max(1, n // cli.sensitivity_states)))
                                           else []) for i in range(n)}
    for i in range(n):
        cell = {"gstar": {}, "path": {}}
        for delta in gstar_deltas[i]:
            probe, span = probes(mu[i:i + 1], delta)  # (1, D, 2, D)
            acts = probe.expand(M, -1, -1, -1).reshape(-1, D)
            seqs = torch.arange(M, device=device).repeat_interleave(2 * D)
            cell["gstar"][delta] = (add(i, seqs, acts).reshape(M, D, 2), span[0].double().cpu().numpy())
        grid_acts = a_fresh[i, :K, None].expand(-1, Mg, -1)
        grid_seqs = (GRID_SEQ + torch.arange(Mg, device=device)).repeat(K)
        cell["grid"] = add(i, grid_seqs, grid_acts).reshape(K, Mg)
        for delta in cli.path_deltas:
            probe, span = probes(a_fresh[i, :K], delta)  # (K, D, 2, D)
            seqs = (GRID_SEQ + torch.arange(K, device=device)).repeat_interleave(2 * D)
            cell["path"][delta] = (add(i, seqs, probe).reshape(K, D, 2), span.double().cpu().numpy())
        cell["extra"] = add(i, EXTRA_SEQ + torch.arange(E, device=device), a_fresh[i, K:], record=True)
        # Smoothed target: d/dmu E_{a ~ pi_mu, xi}[G], the location FD of the whole first-action law, CRN over
        # the first-action draw (same key) and the future (same sequence). This is what score estimators estimate.
        with torch.no_grad():
            loc, span = probes(mu[i:i + 1], cli.deltas[0])  # (1, D, 2, D)
            loc = loc.expand(M, -1, -1, -1).reshape(-1, D)
            keys = base.stream_key(seed_hash, FIRST, i, SMOOTH_FIRST + torch.arange(M, device=device)).repeat_interleave(2 * D)
            acts = policy.sample_shifted(obs0[i:i + 1].expand(len(loc), -1), keys, loc)
        cell["smooth"] = (add(i, (SMOOTH_SEQ + torch.arange(M, device=device)).repeat_interleave(2 * D), acts).reshape(M, D, 2),
                          span[0].double().cpu().numpy())
        layout.append(cell)
    dup = add(0, 0, mu[0:1].expand(2, -1))  # identical branches must give identical returns
    state_index = np.concatenate([np.full(len(p[2]), p[0]) for p in parts])
    branch_states = states[state_index]
    sids = torch.as_tensor(state_index, device=device)
    seqs = torch.cat([p[1] for p in parts])
    first = torch.cat([p[2] for p in parts]).float()
    record = torch.cat([torch.full((len(p[2]),), p[3], device=device) for p in parts])
    for i, cell in enumerate(layout):  # realized samples: grid diagonal
        record[torch.as_tensor(cell["grid"][np.arange(K), np.arange(K)], device=device)] = True
    print(f"branches: {count} x H={H} = {count * H / 1e6:.1f}M env steps", flush=True)
    t_branch = time.perf_counter()
    out = run_branches(policy, physics, branch_states, sids, seqs, first, record, H, cli.chunk, seed_hash, b_phi)
    branch_seconds = time.perf_counter() - t_branch
    gH = policy.gamma ** H
    variants = {"mc": out["mc"], "boot": out["mc"] + out["bonus"] + gH * out["tail"]}
    if policy.temperature > 0.0:
        variants = {"mc_raw": out["mc"], "mc_soft": out["mc"] + out["bonus"], "boot_soft": variants["boot"]}
    variants = {k: v.cpu().numpy() for k, v in variants.items()}
    determinism = {name: float(abs(v[dup[0]] - v[dup[1]])) for name, v in variants.items()}
    assert all(x == 0.0 for x in determinism.values()), f"CRN replay is not deterministic: {determinism}"

    # ---- lambda advantages along recorded trajectories (trainer's lambda; soft rewards for REPPO)
    rec_pos = {int(b): j for j, b in enumerate(out["rec_index"].cpu().numpy())}
    rewards_soft = (out["rec_rewards"] + out["rec_bonus"]).cpu().numpy()
    values = out["rec_values"].cpu().numpy()
    deltas_td = rewards_soft + policy.gamma * values[:, 1:] - values[:, :-1]
    weights = (policy.gamma * policy.lam) ** np.arange(H)
    a_gae = deltas_td @ weights

    # ---- one-step critic gradient (iii) at mu and every a_k, and REPPO's critic dQ/da (iv)
    centers = torch.cat([mu[:, None], a_fresh[:, :K]], 1)  # (n, 1 + K, D)
    onestep = {}
    with torch.no_grad():
        for delta in cli.path_deltas:
            probe, span = probes(centers.reshape(-1, D), delta)  # (n(1+K), D, 2, D)
            acts = probe.reshape(-1, D).float().contiguous()
            reps = probe.shape[0] * D * 2 // n
            st = np.ascontiguousarray(np.repeat(states, reps, axis=0))
            before = torch.from_numpy(st[:, 0].copy()).to(device)
            physics.step(st, acts.cpu().numpy())
            after = torch.from_numpy(st).to(device)
            r = policy.step_reward(policy.raw_reward(after[:, 0], before, acts))
            key = base.stream_key(seed_hash, VALUE, sids_all.repeat_interleave(reps), ONESTEP_SEQ)
            q1 = (r + policy.gamma * policy.value(policy.observe(after), key)).view(n, 1 + K, D, 2)
            onestep[delta] = ((q1[..., 0] - q1[..., 1]) / span.view(n, 1 + K, D).double()).cpu().numpy()
    critic_grad = None
    if policy.critic_action_grad is not None:
        critic_grad = policy.critic_action_grad(obs0[:, None].expand(-1, 1 + K, -1).reshape(-1, obs0.shape[-1]),
                                                centers.reshape(-1, D)).view(n, 1 + K, D).cpu().numpy()

    score_np, fisher_np = score.cpu().numpy(), fisher.cpu().numpy()
    b_phi_np, v_phi_np, mu_np = b_phi.cpu().numpy(), v_phi.cpu().numpy(), mu.double().cpu().numpy()

    meta = {"snapshot": str(path), "kind": kind, "global_step": step, "units": policy.units,
            "trainer": snapshot.get("trainer_module", snapshot.get("trainer_file")),
            "gamma": policy.gamma, "lambda": policy.lam, "temperature": policy.temperature,
            "states": n, "state_times": picks.tolist(), "K": K, "M_grid": Mg, "E_extra": E, "M_gstar": M, "H": H,
            "gamma^H": gH, "deltas": list(cli.deltas), "extra_deltas": list(cli.extra_deltas),
            "branches": count, "env_steps": count * H, "branch_seconds": branch_seconds,
            "crn_duplicate_abs_diff": determinism}
    raw = {"variants": variants, "a_gae": a_gae, "rec_index": out["rec_index"].cpu().numpy(), "score": score_np,
           "fisher": fisher_np, "b_phi": b_phi_np, "v_phi": v_phi_np, "mu": mu_np, "states": states,
           "onestep": onestep, "critic_grad": critic_grad, "layout": layout}
    results = analyze(raw, meta, cli, label)
    meta["seconds"] = time.perf_counter() - start
    return {"meta": meta, "validation": validation, "results": results}, raw


# ---------------------------------------------------------------- reporting

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "inf" if isinstance(x, float) and x == float("inf") else "n/a"
    return f"{x:.{nd}g}" if abs(x) >= 1000 else f"{x:.{nd}f}"


def markdown(reports, cli):
    lines = ["# Credit oracle v1 (HalfCheetah-v4)", "",
             "Generated by `scripts/credit_oracle.py`. Ground truth by CRN branching of the deterministic simulator; "
             "see the script docstring for definitions. cos1 = mean single-sample cosine with g* (corrected for g*'s SE); "
             "cos_inf = cosine of the estimator's expectation; N0.9 = samples at one state for the average to reach "
             "cosine 0.9 (pooled typical state). Natural = inner product with the location Fisher inverse.", ""]
    head = ["score_mc_critic_baseline", "score_gae", "score_oracle_A", "pathwise_hindsight_d0.2",
            "onestep_critic_at_mu_d0.2", "critic_dQda_at_mu"]
    lines += ["## Headline (critic-consistent variant, plain metric, target g* at delta=0.2)", "",
              "N0.9 ~ 4.26 / cos1^2 extrapolates an unbiased per-sample estimator (cos_N = 1/sqrt(1 + (1/cos1^2 - 1)/N)); "
              "deterministic estimators (one-step, critic dQ/da) cannot improve by averaging (their cos_inf = cos1).", "",
              "| snapshot | action / common luck / interaction / critic share | corr(A_mc, A_true) | corr(A_gae, A_true) | "
              + " | ".join(head) + " |", "|-" * (4 + len(head)) + "|"]
    for rep in reports:
        vname = next(v for v in rep["results"] if v.startswith("boot"))
        res = rep["results"][vname]
        sh, fid = res["decomposition"]["shares_of_E[A_mc^2]"], res["decomposition"]["advantage_fidelity"]
        cells = []
        for name in head:
            v = res["estimators"].get(f"{name}|plain|g*d{rep['meta']['deltas'][0]}")
            if v is None:
                cells.append("-")
                continue
            c = v["energy_weighted"]["cos1"]
            extrap = f", N0.9~{4.26 / c ** 2:.0f}" if c > 0 and not name.startswith(("onestep", "critic")) else ""
            cells.append(f"{c:+.3f}{extrap}")
        lines.append(f"| {rep['meta']['kind']} {rep['meta']['global_step']} ({vname}) | {sh['action']:.3f} / {sh['luck_common']:.3f} / "
                     f"{sh['luck_interaction']:.3f} / {sh['critic_baseline']:.3f} | {fid['corr_mc_vs_true']:.3f} | "
                     f"{fid['corr_gae_vs_true']:.3f} | " + " | ".join(cells) + " |")
    lines.append("")
    for rep in reports:
        m = rep["meta"]
        lines += [f"## {m['kind']} @ {m['global_step']}", "",
                  f"`{m['snapshot']}`; units: {m['units']}; N={m['states']} states, K={m['K']}, M_grid={m['M_grid']}, "
                  f"E={m['E_extra']}, M(g*)={m['M_gstar']}, H={m['H']} (gamma^H={m['gamma^H']:.3f}); "
                  f"{m['env_steps'] / 1e6:.1f}M branch steps; episode return {rep['validation']['episode_return_raw']['mean']:.0f}.", ""]
        for vname, res in rep["results"].items():
            d = res["decomposition"]
            sh = d["shares_of_E[A_mc^2]"]
            fid = d["advantage_fidelity"]
            consistent = vname.startswith("boot")
            lines += [f"### variant `{vname}`" + ("" if consistent else
                      " (critic-based rows compare the critic to a different objective: no tail / no entropy terms)"), "",
                      f"- E[A_mc^2] shares: action {sh['action']:.3f}, luck common {sh['luck_common']:.3f}, "
                      f"luck interaction {sh['luck_interaction']:.3f}, critic baseline {sh['critic_baseline']:.3f} "
                      f"(observed E[A^2] {d['E[A_mc^2]_observed']:.4g} vs sum {d['E[A_mc^2]_sum_of_terms']:.4g}); "
                      f"median within-state action share {d['action_share_within_state_median']:.3f}",
                      f"- critic V error: mean {d['critic_V_error']['mean']:+.4g}, rms {d['critic_V_error']['rms_corrected']:.4g}, "
                      f"across-state var / mean within-state Var(G) {d['critic_V_error']['var_across_states_over_mean_within_state_var_A']:.3f}",
                      f"- corr with A_true: MC {fid['corr_mc_vs_true']:.3f} (within-state {fid['corr_within_state_mc_vs_true']:.3f}), "
                      f"GAE {fid['corr_gae_vs_true']:.3f} (within {fid['corr_within_state_gae_vs_true']:.3f}); "
                      f"error share MC {fid['E[(A_mc-A_true)^2]/E[A_mc^2]']:.3f}, GAE {fid['E[(A_gae-A_true)^2]/E[A_gae^2]']:.3f}",
                      f"- g* median rel SE {res['gstar']['rel_se_median']:.2f}, gsmooth {res['gstar']['gsmooth_rel_se_median']:.2f}, "
                      f"median cos(g*, gsmooth) {res['gstar']['cos_gstar_vs_gsmooth_median']:.3f}; delta sensitivity (cos vs delta={cli.deltas[0]}): "
                      + ", ".join(f"{k}: {v['cos_median']:.2f} (|g| ratio {v['norm_ratio_median']:.2f}, n={v['states']})"
                                  for k, v in res["gstar"]["delta_sensitivity"].items()), "",
                      f"- split-half cosine (M/2 vs M/2) median: g* {res['gstar']['gstar_split_half_cos_median']:.3f}, "
                      f"gsmooth {res['gstar']['gsmooth_split_half_cos_median']:.3f}", "",
                      f"- robust: median-state shares action {d['action_share_within_state_median']:.3f}, common luck "
                      f"{d['common_luck_share_within_state_median']:.3f}, critic baseline {d['critic_baseline_share_median_state']:.3f}; "
                      f"top-3 states hold {d['critic_baseline_sq_top3_state_fraction']:.2f} of baseline error^2 and "
                      f"{d['within_state_var_top3_state_fraction']:.2f} of within-state Var(G)", "",
                      "Energy-weighted over all states (90% state-bootstrap CI); `raw` = mean cosine against the noisy g*.", "",
                      "| estimator | metric | target | cos raw | cos1 [CI] | cos_inf [CI] | N0.9 [CI] |",
                      "|-|-|-|-|-|-|-|"]
            for key, v in res["estimators"].items():
                name, metric, target = key.split("|")
                ew = v["energy_weighted"]
                lines.append(f"| {name} | {metric} | {target} | {fmt(v.get('cos_raw_mean'))} | {fmt(ew['cos1'])} "
                             f"[{fmt(ew['cos1_ci90'][0], 2)}, {fmt(ew['cos1_ci90'][1], 2)}] | {fmt(ew['cos_inf'])} "
                             f"[{fmt(ew['cos_inf_ci90'][0], 2)}, {fmt(ew['cos_inf_ci90'][1], 2)}] | {fmt(ew['n09'])} "
                             f"[{fmt(ew['n09_ci90'][0], 2)}, {fmt(ew['n09_ci90'][1], 2)}] |")
            lines.append("")
    return "\n".join(lines)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "per_state":
                continue
            yield from flatten(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix.replace("|", "/").replace("^", "").replace("*", "star").replace("[", "").replace("]", ""), value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", type=Path, nargs="*", default=[])
    p.add_argument("--reppo", type=Path, nargs="*", default=[])
    p.add_argument("--states", type=int, default=32)
    p.add_argument("--skip", type=int, default=50)
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--rollouts", type=int, default=256, help="M: CRN sequences for g* and gsmooth")
    p.add_argument("--grid-rollouts", type=int, default=24, help="M_g: sequences per on-policy action in the grid")
    p.add_argument("--samples", type=int, default=16, help="K: on-policy actions per state (grid rows, realized)")
    p.add_argument("--extra", type=int, default=48, help="E: extra independent realized rollouts for score estimators")
    # At M=32, |g*_delta| scaled ~1/delta over 0.02..0.2 and g*_0.05 vs g*_0.2 had cosine ~0: CRN differences of
    # 400-step HalfCheetah returns decorrelate (chaos), so small-delta FDs are pure noise. Hence delta=0.2, M=256.
    p.add_argument("--deltas", type=float, nargs="+", default=[0.2], help="g* targets; the first is primary")
    p.add_argument("--path-deltas", type=float, nargs="+", default=[0.2, 0.05], help="pathwise / one-step FD steps")
    p.add_argument("--extra-deltas", type=float, nargs="*", default=[0.1, 0.4])
    p.add_argument("--sensitivity-states", type=int, default=4)
    p.add_argument("--value-samples", type=int, default=16)
    p.add_argument("--baseline-samples", type=int, default=4096, help="REPPO E_pi Q_phi(s, .) draws at the selected states")
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--norm-row", type=int, default=0)
    p.add_argument("--validate-episodes", type=int, default=16)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--chunk", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=REPO / "benchmarks/credit_oracle_v1")
    p.add_argument("--rescore", nargs="*", default=None, help="tags (e.g. control_4013696): recompute statistics "
                   "from runs/credit_oracle_v1/<tag>_raw.pt and <out-dir>/<tag>.json, no simulation")
    cli = p.parse_args()
    if cli.rescore is not None:
        reports = []
        for tag in cli.rescore:
            report = json.loads((cli.out_dir / f"{tag}.json").read_text())
            raw = torch.load(REPO / "runs/credit_oracle_v1" / f"{tag}_raw.pt", weights_only=False)
            cli.deltas = report["meta"]["deltas"]
            report["results"] = analyze(raw, report["meta"], cli, tag)
            (cli.out_dir / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float))
            reports.append(report)
        (cli.out_dir / "summary.md").write_text(markdown(reports, cli))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit through mlq")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = REPO / "runs/credit_oracle_v1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for kind, paths in (("control", cli.control), ("reppo", cli.reppo)):
        for path in paths:
            torch.manual_seed(cli.seed)
            report, raw = run_snapshot(kind, path, cli, device)
            reports.append(report)
            tag = f"{kind}_{report['meta']['global_step']}"
            (cli.out_dir / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float))
            torch.save(raw, raw_dir / f"{tag}_raw.pt")
            with SummaryWriter(str(REPO / "runs" / f"{ENV_ID}__credit_oracle_v1_{tag}__{cli.seed}__{int(time.time())}")) as w:
                for name, value in flatten(report["results"]):
                    w.add_scalar(f"oracle/{name}", value, report["meta"]["global_step"])
            (cli.out_dir / "summary.md").write_text(markdown(reports, cli))
            print(f"saved {cli.out_dir / tag}.json", flush=True)
    (cli.out_dir / "summary.md").write_text(markdown(reports, cli))


if __name__ == "__main__":
    main()
