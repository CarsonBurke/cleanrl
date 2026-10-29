"""How accurate are a REPPO critic's action gradient and advantages against the true soft Q^pi?

    python scripts/reppo_oracle.py SNAPSHOT.pt [SNAPSHOT.pt ...] [--states 256 --rollouts 8 --horizon 300]
    python scripts/reppo_oracle.py SNAPSHOT.pt --states 512 --rollouts 32 --fd-steps 0.2 0.05   # the reported runs
    python scripts/reppo_oracle.py --rescore runs/reppo_oracle/DATASET.pt   # statistics only, from a saved dataset

Snapshots come from cleanrl/reppo/ppo_continuous_action_reppo_probe_v23.py --snapshot-steps (HalfCheetah-v4).
The trainer module is imported from its path to rebuild Actor and Critic from the snapshot's args.

Ground truth, in the trainer's scaled units (reward x reward_scale, temperature T from the snapshot):

    Q(s0, a0) = r0 + sum_{k>=1} gamma^k (r_k - T log pi(a_k | s_k)),  a_k ~ pi (tanh-Gaussian) for k >= 1,

measured by branching restored MuJoCo states with scripts/awr_oracle_advantage.py's machinery: its C kernel
(validated bitwise against gymnasium here too) and its counter-based hash, from which the tanh-Gaussian
noise is drawn as normals (inverse CDF), so rollout m of every action at a state sees the same future noise
(common random numbers). The 1000-step time limit bootstraps as the trainer does, and so does the horizon:
gamma^t V(s_t), V(s) = E_{a~pi}[Q_critic(s, a) - T log pi(a | s)] from --value-samples CRN draws; partial
results are reported at every --checkpoints horizon.

On --states states drawn from one on-policy 16 x 1024 chunk (staggered starting phases, like the trainer),
each state's action set is: the chunk's sampled action a0 and its 2D finite-difference probes, the mean
action a* = tanh(mu) and its probes, and --fresh actions a_k ~ pi. Measured:

1. dQ/da at a* and a0 by central differences at every --fd-steps h (one-sided where a +- h leaves
   [-1, 1]); per-component SEs from the spread over m of the CRN-paired differences. CRN barely helps
   here: a 0.05 action change decorrelates HalfCheetah's rollouts within 100 steps, and Q^pi is rough at
   small action scales, so h ~ the policy's action spread (0.2) is both the more precise and the more
   relevant (smoothed) gradient; the agreement between the first two steps is reported.
2. The critic's autograd dQ/da (and its own finite difference at the same h) against it: mean per-state
   cosine; a pooled cosine mean_s <c_hat, g> / sqrt(mean_s |g_true|^2), with |g_true|^2 estimated by
   |g|^2 - sum SE^2 ("noise-corrected") or by the split-half product <g_A, g_B>; the norm ratio; and the
   cosine of the batch-mean directions. The pooled cosine is energy-weighted (a perfect critic scores
   mean|g| / rms|g| < 1), and the true-gradient energy is heavy-tailed (near-flip states), so every
   statistic is also reported on the upright and cruising subsets (see subsets()).
3. Within-state advantage fidelity over the fresh actions (leave-one-out A as in awr_oracle): pooled
   correlation, regression slopes, and the share of the (noise-corrected) true within-state variance the
   critic explains.
4. The trainer's lambda-return advantage G - E_pi Q_critic(s, .) on a0 (targets bootstrapped by the
   snapshot critic exactly as the trainer builds them) against A_true(a0), next to the critic's own.
5. A dataset under --out-dir: states, env rows, ages, actions, oracle returns (every checkpoint and
   rollout), derived labels and SEs, and the full chunk's transitions for offline critic experiments.

Run it through mlq (CUDA). Physics dominates: n (4D + 2 + K) M H env steps (D = 6).
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awr_oracle_advantage as base  # noqa: E402  (its bench import puts the repo root on sys.path)

from cleanrl.shared.ppo_loop import get_gae_fn  # noqa: E402
from cleanrl.shared.staggered_envs import compute_phase_offsets  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "cleanrl" / "reppo" / "ppo_continuous_action_reppo_probe_v23.py"
ENV_ID, TIME_LIMIT = base.ENV_ID, base.TIME_LIMIT
# Counter-based streams (disjoint from each other by the first hash part).
COLLECT, FRESH, BRANCH, TEST, VALIDATE, BOOT, TARGET, BASELINE = 1, 2, 3, 4, 5, 6, 7, 8


def load_trainer():
    spec = importlib.util.spec_from_file_location("reppo_probe_v23", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- counter-based tanh-Gaussian

def normals(key, count):
    """(..., count) standard normals, draw j = Phi^-1(hash(key, j)) (24-bit uniforms, |z| <= 5.3).

    base.uniforms' bits, but the half-offset uniform is formed in float64: in float32 the top word
    rounds to exactly 1 and erfinv(1) = inf (one draw in 2^24)."""
    bits = base.mix32(base.fold(key[..., None], torch.arange(count, device=key.device)))
    u = ((bits >> 8).double() + 0.5) * 2.0 ** -24
    return (math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)).float()


def squash(mean, std, noise):
    """The trainer's squashed_sample with explicit noise: tanh(mean + std noise) and its log-density."""
    pre = mean + std * noise
    log_det = 2.0 * (math.log(2.0) - pre - F.softplus(-2.0 * pre))
    logprob = (-0.5 * noise.square() - std.log() - 0.5 * math.log(2.0 * math.pi) - log_det).sum(-1)
    return torch.tanh(pre), logprob


def validate_sampler(trainer, device, seed_hash):
    """Hash vs a uint32 reference, normal marginals (KS), and squash == the trainer's squashed_sample."""
    words = np.random.default_rng(0).integers(0, 2 ** 32, 100_000, dtype=np.uint64)
    reference = words.astype(np.uint32)
    with np.errstate(over="ignore"):
        for shift, constant in ((16, 0x7FEB352D), (15, 0x846CA68B)):
            reference ^= reference >> np.uint32(shift)
            reference *= np.uint32(constant)
        reference ^= reference >> np.uint32(16)
    ours = base.mix32(torch.as_tensor(words.astype(np.int64), device=device)).cpu().numpy()
    assert np.array_equal(ours, reference.astype(np.int64)), "mix32 disagrees with the uint32 reference"
    draws = normals(base.stream_key(seed_hash, TEST, torch.arange(100_000, device=device)), 4).double().cpu().numpy()
    ks = stats.kstest(draws.ravel(), "norm")
    assert ks.pvalue > 1e-4, f"hash normals fail KS: {ks}"
    corr = np.corrcoef(draws.T)
    assert np.abs(corr - np.eye(4)).max() < 0.02, "hash normals are correlated across draws"
    generator = torch.Generator(device=device).manual_seed(0)
    mean = torch.randn((4096, 6), device=device, generator=generator) * 2.0
    std = torch.rand((4096, 6), device=device, generator=generator) + 0.05
    torch.manual_seed(123)
    action_ref, logprob_ref = trainer.squashed_sample(mean, std)
    torch.manual_seed(123)
    action, logprob = squash(mean, std, torch.randn_like(mean))
    assert torch.equal(action, action_ref) and torch.allclose(logprob, logprob_ref, atol=1e-5), "squash != trainer"
    return {"ks": float(ks.statistic), "ks_p": float(ks.pvalue), "max_offdiag_corr": float(np.abs(corr - np.eye(4)).max())}


# ---------------------------------------------------------------- frozen snapshot

class Oracle:
    """Frozen actor + critic (CUDA) with the snapshot's per-env observation normalization."""

    def __init__(self, trainer, snapshot, physics, cli, device):
        args = SimpleNamespace(**snapshot["args"])
        self.args, self.physics, self.device = args, physics, device
        obs_dim, self.action_dim = physics.nq - 1 + physics.nv, physics.nu
        self.actor = trainer.Actor(obs_dim, self.action_dim, args).to(device)
        self.actor.net.load_state_dict(snapshot["actor_net"])
        self.critic = trainer.Critic(obs_dim, self.action_dim, args, device).to(device)
        self.critic.load_state_dict(snapshot["critic"])
        for module in (self.actor, self.critic):
            module.eval().requires_grad_(False)
        self.gamma, self.lam, self.reward_scale = float(args.gamma), float(args.gve_lambda), float(args.reward_scale)
        self.temperature, self.obs_clip = float(snapshot["temperature"]), float(args.obs_clip)
        as64 = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)  # noqa: E731
        self.obs_mean = as64(snapshot["obs_means"])
        self.obs_scale = torch.sqrt(as64(snapshot["obs_variances"]) + float(snapshot["obs_epsilon"]))
        self.num_envs = self.obs_mean.shape[0]
        self.value_samples = cli.value_samples
        self.seed_hash = int(base.mix32(cli.seed + 0x5EED))

    def observe(self, state, rows):
        obs = state[:, 1:self.physics.nq + self.physics.nv]
        return ((obs - self.obs_mean[rows]) / self.obs_scale[rows]).clamp(-self.obs_clip, self.obs_clip).float()

    def raw_reward(self, x_after, x_before, action):
        forward = base.FORWARD_REWARD_WEIGHT * ((x_after - x_before) / self.physics.dt)
        return forward - base.CTRL_COST_WEIGHT * action.double().square().sum(-1)

    def moments(self, obs):
        return self.actor(obs)

    def sample(self, obs, key):
        """Tanh-Gaussian actions and log-probs from the per-row hash keys."""
        mean, std = self.moments(obs)
        return squash(mean, std, normals(key, self.action_dim))

    def soft_value(self, obs, key, chunk=8192):
        """V(s) = mean_j [Q_critic(s, a_j) - T log pi(a_j | s)], a_j from hash(key, j): CRN across branches."""
        out = []
        samples = torch.arange(self.value_samples, device=self.device)
        for start in range(0, obs.shape[0], chunk):
            o, k = obs[start:start + chunk], key[start:start + chunk]
            mean, std = self.moments(o)
            noise = normals(base.fold(k[:, None], samples), self.action_dim)  # (n, S, D)
            action, logprob = squash(mean[:, None], std[:, None], noise)
            q = self.critic.value(o[:, None].expand(-1, self.value_samples, -1), action)
            out.append((q - self.temperature * logprob).mean(1))
        return torch.cat(out).double()

    def critic_q(self, obs, actions):
        """Q_critic(s, a) for actions (n, A, D) -> (n, A)."""
        return self.critic.value(obs[:, None].expand(-1, actions.shape[1], -1), actions)

    def critic_grad(self, obs, actions):
        with torch.enable_grad():
            actions = actions.detach().clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(self.critic.value(obs, actions).sum(), actions)
        return grad


# ---------------------------------------------------------------- validation

@torch.no_grad()
def simulator_returns(oracle, episodes):
    physics, device = oracle.physics, oracle.device
    state = physics.reset(episodes)
    lanes = torch.arange(episodes, device=device)
    rows = lanes % oracle.num_envs
    total = torch.zeros(episodes, dtype=torch.float64, device=device)
    for step in range(TIME_LIMIT):
        current = torch.from_numpy(state).to(device)
        action, _ = oracle.sample(oracle.observe(current, rows), base.stream_key(oracle.seed_hash, VALIDATE, lanes, step))
        physics.step(state, action.cpu().numpy())
        total += oracle.raw_reward(torch.from_numpy(state[:, 0]).to(device), current[:, 0], action)
    return total.cpu().numpy()


@torch.no_grad()
def gymnasium_returns(oracle, episodes, seed):
    """The frozen policy in plain gymnasium envs with numpy noise (independent of the kernel and hash)."""
    envs = [gym.make(ENV_ID) for _ in range(episodes)]
    obs = np.stack([env.reset(seed=seed + 20_000 + i)[0] for i, env in enumerate(envs)])
    rows = torch.arange(episodes, device=oracle.device) % oracle.num_envs
    rng = np.random.default_rng(seed + 20_000)
    total = np.zeros(episodes)
    for step in range(TIME_LIMIT):
        normalized = ((torch.as_tensor(obs, dtype=torch.float64, device=oracle.device) - oracle.obs_mean[rows])
                      / oracle.obs_scale[rows]).clamp(-oracle.obs_clip, oracle.obs_clip).float()
        mean, std = oracle.moments(normalized)
        noise = torch.as_tensor(rng.standard_normal(mean.shape, dtype=np.float32), device=oracle.device)
        action = torch.tanh(mean + std * noise).cpu().numpy()
        for i, env in enumerate(envs):
            obs[i], reward, terminated, truncated, _ = env.step(action[i])
            total[i] += reward
            assert not terminated and truncated == (step + 1 == TIME_LIMIT)
    for env in envs:
        env.close()
    return total


def validate(oracle, trainer, cli):
    start = time.perf_counter()
    out = {"sampler": validate_sampler(trainer, oracle.device, oracle.seed_hash),
           "physics": base.validate_physics(oracle, cli.seed)}
    ours, theirs = simulator_returns(oracle, cli.validate_episodes), gymnasium_returns(oracle, cli.gym_episodes, cli.seed)
    z = (ours.mean() - theirs.mean()) / math.sqrt(ours.var(ddof=1) / len(ours) + theirs.var(ddof=1) / len(theirs))
    out["returns"] = {"simulator_mean": float(ours.mean()), "simulator_sd": float(ours.std()), "simulator_n": len(ours),
                      "gymnasium_mean": float(theirs.mean()), "gymnasium_sd": float(theirs.std()),
                      "gymnasium_n": len(theirs), "z": float(z)}
    print(f"validation: hash normals KS {out['sampler']['ks']:.4f} (p {out['sampler']['ks_p']:.2f}), squash == trainer; "
          f"physics bitwise vs gymnasium over {TIME_LIMIT} steps x {out['physics']['copies']} copies, reward err "
          f"{out['physics']['max_reward_error']:.1e}, {out['physics']['env_steps_per_s'] / 1e6:.2f}M steps/s; returns "
          f"{ours.mean():.1f}+-{ours.std():.1f} (n {len(ours)}) vs gymnasium {theirs.mean():.1f}+-{theirs.std():.1f} "
          f"(n {len(theirs)}) z={z:+.2f} [{time.perf_counter() - start:.0f}s]", flush=True)
    assert abs(z) < 4.0, f"frozen-policy return mismatch: z={z:.2f}"
    return out


# ---------------------------------------------------------------- on-policy chunk

@torch.no_grad()
def collect(oracle, trainer, num_envs, num_steps, seed):
    """One trainer-shaped chunk from staggered phases, with the trainer's soft lambda-return targets."""
    physics, device = oracle.physics, oracle.device
    state = physics.reset(num_envs)
    age = np.zeros(num_envs, dtype=np.int64)
    rows = torch.arange(num_envs, device=device)
    clock = 0

    def act():
        nonlocal clock
        current = torch.from_numpy(state).to(device)
        action, logprob = oracle.sample(oracle.observe(current, rows), base.stream_key(oracle.seed_hash, COLLECT, rows, clock))
        clock += 1
        return current, action, logprob

    offsets = compute_phase_offsets(num_envs, TIME_LIMIT, seed)
    for step in range(int(offsets.max())):  # stochastic warmup to the staggered ages
        stepped = state.copy()
        physics.step(stepped, act()[1].cpu().numpy())
        active = offsets > step
        state[active], age[active] = stepped[active], age[active] + 1
    shape = (num_steps, num_envs)
    states = np.empty(shape + (physics.width,))
    ages = np.empty(shape, dtype=np.int64)
    next_raw = torch.empty(shape + (physics.nq - 1 + physics.nv,), dtype=torch.float64, device=device)
    actions = torch.empty(shape + (physics.nu,), device=device)
    logprobs = torch.empty(shape, device=device)
    rewards = torch.empty(shape, device=device)
    truncations = torch.zeros(shape, device=device)
    for step in range(num_steps):
        current, action, logprob = act()
        states[step], ages[step] = state, age
        physics.step(state, action.cpu().numpy())
        after = torch.from_numpy(state).to(device)
        rewards[step] = oracle.reward_scale * oracle.raw_reward(after[:, 0], current[:, 0], action)
        next_raw[step] = after[:, 1:physics.nq + physics.nv]  # the final observation before any reset
        actions[step], logprobs[step] = action, logprob
        age += 1
        done = np.flatnonzero(age >= TIME_LIMIT)
        if done.size:
            truncations[step, torch.as_tensor(done, device=device)] = 1.0
            state[done], age[done] = physics.reset(done.size), 0
    # The trainer's targets: Q'_t = Q(s_{t+1}, a'), a' ~ pi fresh; soft reward r - gamma T log pi(a' | s');
    # boundaries at truncations and the chunk end; no terminations in HalfCheetah.
    flat_rows = rows.repeat(num_steps)
    sids = torch.arange(num_steps * num_envs, device=device)
    next_obs = ((next_raw.flatten(0, 1) - oracle.obs_mean[flat_rows]) / oracle.obs_scale[flat_rows]).clamp(
        -oracle.obs_clip, oracle.obs_clip).float()
    next_action, next_logprob = oracle.sample(next_obs, base.stream_key(oracle.seed_hash, TARGET, sids))
    next_q = oracle.critic.value(next_obs, next_action).view(shape)
    next_logprob = next_logprob.view(shape)
    soft_rewards = rewards - oracle.gamma * oracle.temperature * next_logprob
    ends = torch.zeros(shape, device=device)
    ends[-1] = 1.0
    targets = trainer.lambda_returns(get_gae_fn(compiled=False, explicit_next_values=True), soft_rewards, next_q,
                                     torch.zeros(shape, device=device), torch.maximum(truncations, ends),
                                     oracle.gamma, oracle.lam)
    return {"state": states.reshape(-1, physics.width), "age": ages.ravel(), "row": np.tile(np.arange(num_envs), num_steps),
            "raw_obs": torch.as_tensor(states.reshape(-1, physics.width)[:, 1:physics.nq + physics.nv]),
            "raw_next_obs": next_raw.flatten(0, 1).cpu(), "actions": actions.flatten(0, 1).cpu(),
            "logprob": logprobs.flatten().cpu(), "rewards": rewards.flatten().cpu(),
            "truncations": truncations.flatten().cpu(), "ends": ends.flatten().cpu(), "next_q": next_q.flatten().cpu(),
            "next_logprob": next_logprob.flatten().cpu(), "targets": targets.flatten().cpu(),
            "num_envs": num_envs, "num_steps": num_steps}


# ---------------------------------------------------------------- branching

def probes(center, steps):
    """(n, 1 + 2D len(steps), D) = [a, then per h: a + h e_1.., a - h e_1..] (the centre where a probe
    would leave [-1, 1]) and, per h, the (n, D) distance between each plus and minus probe."""
    count, dims = center.shape
    out, spans = [center[:, None]], []
    for h in steps:
        eye = torch.eye(dims, device=center.device) * h
        plus = torch.where((center + h <= 1.0)[:, :, None], center[:, None] + eye, center[:, None])
        minus = torch.where((center - h >= -1.0)[:, :, None], center[:, None] - eye, center[:, None])
        out += [plus, minus]
        spans.append(torch.diagonal(plus - minus, dim1=1, dim2=2))
    return torch.cat(out, 1), spans


class BranchGroup:
    """All (state, action, rollout) branches of a group of states, stepped in lockstep."""

    def __init__(self, oracle, cli, states, ages, rows, sids, actions):
        count, per_state, _ = actions.shape
        device, self.oracle, self.rollouts = oracle.device, oracle, cli.rollouts
        per = per_state * cli.rollouts
        self.shape = (count, per_state, cli.rollouts)
        self.state = np.ascontiguousarray(np.repeat(states, per, axis=0))
        expand = lambda values: torch.as_tensor(values, device=device).repeat_interleave(per)  # noqa: E731
        self.ages, self.rows, self.sids = expand(ages), expand(rows), expand(sids)
        self.rollout = torch.arange(cli.rollouts, device=device).repeat(count * per_state)
        self.action = actions.reshape(-1, actions.shape[-1]).repeat_interleave(cli.rollouts, 0).contiguous()
        self.ctrl = self.action.cpu().numpy()
        self.x_before = expand(states[:, 0])
        self.total = torch.zeros(self.ages.shape[0], dtype=torch.float64, device=device)
        self.alive = torch.ones_like(self.total, dtype=torch.bool)
        self.returns = {}

    def advance(self, t, horizon, checkpoints):
        """Consume physics step t - 1 (taken with self.action), then draw the step-t action."""
        oracle, gamma = self.oracle, self.oracle.gamma
        current = torch.from_numpy(self.state).to(oracle.device)
        raw = oracle.raw_reward(current[:, 0], self.x_before, self.action)
        self.total += torch.where(self.alive, gamma ** (t - 1) * oracle.reward_scale * raw, 0.0)
        obs = oracle.observe(current, self.rows)
        truncated = self.alive & (self.ages + t >= TIME_LIMIT)
        stop = torch.nonzero(truncated).flatten() if t not in checkpoints else None
        key = base.stream_key(oracle.seed_hash, BOOT, self.sids, self.rollout, t)
        if t in checkpoints:
            value = oracle.soft_value(obs, key)
            self.total += torch.where(truncated, gamma ** t * value, 0.0)
            self.alive &= ~truncated
            self.returns[t] = (self.total + torch.where(self.alive, gamma ** t * value, 0.0)).view(self.shape).cpu().numpy()
        elif stop.numel():
            self.total[stop] += gamma ** t * oracle.soft_value(obs[stop], key[stop])
            self.alive &= ~truncated
        if t < horizon:
            self.action, logprob = oracle.sample(obs, base.stream_key(oracle.seed_hash, BRANCH, self.sids, self.rollout, t))
            self.total -= torch.where(self.alive, gamma ** t * oracle.temperature * logprob.double(), 0.0)
            self.x_before = current[:, 0].clone()
            self.ctrl = self.action.cpu().numpy()


@torch.no_grad()
def oracle_returns(oracle, cli, data, index, actions, pool):
    """{H: (n, A, M) float64} soft Q returns for the chosen states and their action sets.

    All groups are in flight: while one group's physics runs on the pool, the main thread does the
    others' GPU work (ctypes drops the GIL)."""
    checkpoints = sorted(set(cli.checkpoints) | {cli.horizon})
    groups = [BranchGroup(oracle, cli, data["state"][part], data["age"][part], data["row"][part], part, actions[start:start + cli.group_states])
              for start in range(0, len(index), cli.group_states) for part in [index[start:start + cli.group_states]]]
    begin = time.perf_counter()
    pending = [pool.submit(oracle.physics.step, group.state, group.ctrl) for group in groups]
    for t in range(1, cli.horizon + 1):
        for g, group in enumerate(groups):
            pending[g].result()
            group.advance(t, cli.horizon, checkpoints)
            if t < cli.horizon:
                pending[g] = pool.submit(oracle.physics.step, group.state, group.ctrl)
        if t in checkpoints:
            elapsed = time.perf_counter() - begin
            steps = sum(group.state.shape[0] for group in groups) * t
            print(f"  oracle: t={t}/{cli.horizon}, {elapsed:.0f}s, {steps / elapsed / 1e6:.2f}M env steps/s", flush=True)
    return {h: np.concatenate([group.returns[h] for group in groups]) for h in checkpoints}


# ---------------------------------------------------------------- statistics

def finite_difference(returns, span, which):
    """Per-rollout paired central differences (n, D, M) for step `which` of a probe block (n, 1 + 2DJ, M)."""
    dims = span.shape[1]
    plus = 1 + 2 * dims * which
    return (returns[:, plus:plus + dims] - returns[:, plus + dims:plus + 2 * dims]) / span[:, :, None]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cosine(a, b):
    return (unit(a) * unit(b)).sum(-1)


def gradient_stats(per_rollout, critic, critic_fd, rng, boots=500):
    """Critic gradient c (n, D) against the oracle g = mean_m of per_rollout (n, D, M)."""
    count, dims, rollouts = per_rollout.shape
    g = per_rollout.mean(-1)
    se2 = per_rollout.var(-1, ddof=1) / rollouts
    half = rollouts // 2
    g_a, g_b = per_rollout[..., :half].mean(-1), per_rollout[..., half:].mean(-1)
    dot = (unit(critic) * g).sum(-1)
    energy, noise = (g * g).sum(-1), se2.sum(-1)

    def pooled(sel):
        signal = (energy[sel] - noise[sel]).mean()
        return dot[sel].mean() / math.sqrt(signal) if signal > 0 else float("nan")

    everything = np.arange(count)
    boot = [pooled(rng.integers(0, count, count)) for _ in range(boots)]
    signal = (energy - noise).mean()
    mean_c, mean_g = critic.mean(0), g.mean(0)
    mean_noise = se2.sum(0).sum() / count ** 2
    mean_signal = (mean_g ** 2).sum() - mean_noise
    return {
        "cos_mean": float(cosine(critic, g).mean()),
        "cos_pooled_raw": float(dot.mean() / math.sqrt(energy.mean())),
        "cos_pooled_corrected": float(pooled(everything)),
        "cos_pooled_corrected_se": float(np.nanstd(boot)),
        "cos_pooled_splithalf": float(dot.mean() / math.sqrt(max((g_a * g_b).sum(-1).mean(), 1e-30))),
        "oracle_splithalf_cos_mean": float(cosine(g_a, g_b).mean()),
        "oracle_snr": float(signal / noise.mean()),  # true energy over noise energy, per state on average
        "norm_ratio": float(math.sqrt((critic ** 2).sum(-1).mean() / signal)) if signal > 0 else float("nan"),
        "batch_cos_raw": float(cosine(mean_c, mean_g)),
        "batch_cos_corrected": float(unit(mean_c) @ mean_g / math.sqrt(mean_signal)) if mean_signal > 0 else float("nan"),
        "critic_fd_cos_mean": float(cosine(critic_fd, g).mean()),
        "critic_fd_cos_pooled_corrected": float((unit(critic_fd) * g).sum(-1).mean() / math.sqrt(signal)) if signal > 0 else float("nan"),
        "critic_autograd_vs_fd_cos": float(cosine(critic, critic_fd).mean()),
        "true_grad_rms": float(math.sqrt(max(signal, 0.0) / dims)),
        # Heavy tails: the share of the summed true-gradient energy held by the top 1% / 5% of states.
        "energy_share_top1": float(np.sort(energy - noise)[::-1][:max(1, count // 100)].sum() / (energy - noise).sum()),
        "energy_share_top5": float(np.sort(energy - noise)[::-1][:max(1, count // 20)].sum() / (energy - noise).sum()),
        "oracle_se_rms": float(math.sqrt(se2.mean())),
        "critic_grad_rms": float(math.sqrt((critic ** 2).mean())),
    }


def advantage_stats(true_fresh, critic_fresh):
    """Within-state fidelity over K fresh actions. true_fresh (n, K, M), critic_fresh (n, K)."""
    count, fresh, rollouts = true_fresh.shape
    centered = true_fresh - true_fresh.mean(1, keepdims=True)  # CRN-paired within each rollout index
    t = centered.mean(2)
    c = critic_fresh - critic_fresh.mean(1, keepdims=True)
    cov = (c * t).sum(1) / (fresh - 1)
    var_c = (c * c).sum(1) / (fresh - 1)
    var_t = (t * t).sum(1) / (fresh - 1)
    noise = centered.var(2, ddof=1).sum(1) / (rollouts * (fresh - 1))  # as awr_oracle_advantage
    signal = var_t.mean() - noise.mean()
    corr = cov.mean() / math.sqrt(var_c.mean() * signal) if signal > 0 else float("nan")
    per_state = [np.corrcoef(c[i], t[i])[0, 1] for i in range(count)]
    return {"corr_raw": float(cov.mean() / math.sqrt(var_c.mean() * var_t.mean())), "corr_corrected": float(corr),
            "explained_share": float(corr ** 2) if np.isfinite(corr) else float("nan"),
            "slope_true_on_critic": float(cov.mean() / var_c.mean()),
            "slope_critic_on_true": float(cov.mean() / signal) if signal > 0 else float("nan"),
            "sd_ratio_critic_over_true": float(math.sqrt(var_c.mean() / signal)) if signal > 0 else float("nan"),
            "per_state_corr_mean": float(np.nanmean(per_state)),
            "true_within_sd": float(math.sqrt(max(signal, 0.0))), "noise_share": float(noise.mean() / var_t.mean())}


def sampled_stats(true_a0, true_fresh, critic_a0, critic_baseline, target, within_signal):
    """Sampled-action a0: lambda-return and critic advantages against A_true(a0) = Q(a0) - mean_k Q(a_k).

    The estimates subtract critic_baseline, the critic's E_pi Q(s, .) over draws independent of the K
    oracle actions, so the oracle baseline's own error (variance within_signal / K per state, fixed
    across rollouts) is independent of them and is removed from A_true's reliability with the rollout
    noise."""
    fresh = true_fresh.shape[1]
    per = true_a0 - true_fresh.mean(1)  # (n, M) CRN-paired
    truth, se = per.mean(1), per.std(1, ddof=1) / math.sqrt(per.shape[1])
    noise = np.mean(se ** 2) + max(within_signal, 0.0) / fresh
    reliability = 1.0 - noise / truth.var(ddof=1)
    estimates = {"lambda_return": target - critic_baseline, "critic": critic_a0 - critic_baseline}
    out = {"a_true_sd": float(truth.std()), "a_true_se_rms": float(math.sqrt(np.mean(se ** 2))),
           "baseline_error_sd": float(math.sqrt(max(within_signal, 0.0) / fresh)), "reliability": float(reliability)}
    for name, estimate in estimates.items():
        corr = float(np.corrcoef(estimate, truth)[0, 1])
        out[name] = {"corr": corr, "corr_corrected": corr / math.sqrt(reliability) if reliability > 0 else float("nan"),
                     "slope_true_on_est": float(np.cov(truth, estimate)[0, 1] / estimate.var(ddof=1)),
                     "sd_ratio": float(estimate.std() / truth.std())}
    # Value level: the critic's E_pi Q against the oracle's (state-level error).
    v_true = true_fresh.mean((1, 2))
    out["value"] = {"corr": float(np.corrcoef(critic_baseline, v_true)[0, 1]), "bias": float((critic_baseline - v_true).mean()),
                    "error_sd": float((critic_baseline - v_true).std()), "true_sd": float(v_true.std())}
    out["q_a0"] = {"critic_corr": float(np.corrcoef(critic_a0, true_a0.mean(1))[0, 1]),
                   "target_corr": float(np.corrcoef(target, true_a0.mean(1))[0, 1])}
    return out


def subsets(dataset):
    """State subsets scored separately. The gradient energy is extremely heavy-tailed (a few states near a
    flip hold most of it), so besides all states: upright (cos(root angle) > 0; a flipped cheetah is a
    distinct regime of large negative values) and cruising (upright with V_true >= 0.9 median V_true, i.e.
    away from failure; selected on the fresh actions' value, not on the gradient estimates)."""
    states, meta = dataset["states"], dataset["meta"]
    upright = np.cos(states["state"][:, 2].double().numpy()) > 0.0
    longest = dataset["returns"][str(max(int(h) for h in dataset["returns"]))].double().numpy()
    value = longest[:, meta["action_layout"]["fresh"][0]:].mean((1, 2))
    return {"all": np.ones_like(upright), "upright": upright,
            "cruising": upright & (value >= 0.9 * np.median(value))}


def score(dataset, mask):
    """Every statistic, per checkpoint horizon, over the masked states of a saved dataset (no simulation):
    (results, labels)."""
    meta, states = dataset["meta"], dataset["states"]
    rollouts = int(meta["M"])
    layout = meta["action_layout"]
    first_fresh = layout["fresh"][0]
    pick = lambda value: value.double().numpy()[mask]  # noqa: E731
    critic_q = pick(states["critic_q"])
    target = dataset["chunk"]["targets"][states["index"].long()].double().numpy()[mask]
    critic_baseline = pick(states["critic_baseline"]) if "critic_baseline" in states else critic_q[:, first_fresh:].mean(1)
    results, labels = {}, {}
    for h, q in ((int(h), pick(q)) for h, q in dataset["returns"].items()):
        cell = {}
        for key in ("star", "a0"):
            offset, stop = layout[f"{key}_probes"]
            spans = states[f"span_{key}"].double().numpy()[:, mask]
            critic_grad = pick(states[f"critic_grad_{key}"])
            estimates = []
            for which, step in enumerate(meta["fd_steps"]):
                per_rollout = finite_difference(q[:, offset:stop], spans[which], which)
                critic_fd = finite_difference(critic_q[:, offset:stop, None], spans[which], which)[..., 0]
                name = f"{key}_h{step:g}"
                cell[f"grad_{name}"] = gradient_stats(per_rollout, critic_grad, critic_fd, np.random.default_rng(h))
                labels[f"grad_{name}_{h}"] = per_rollout.mean(-1)
                labels[f"grad_{name}_se_{h}"] = per_rollout.std(-1, ddof=1) / math.sqrt(rollouts)
                estimates.append(per_rollout)
            if len(estimates) > 1:
                # Finite-difference scale: do the true gradients at the first two steps agree? Both use the
                # same rollouts (CRN), so their noise covariance is removed from the cross term too.
                e1, e2 = estimates[:2]
                g1, g2 = e1.mean(-1), e2.mean(-1)
                cross_noise = ((e1 - g1[..., None]) * (e2 - g2[..., None])).sum(-1) / ((rollouts - 1) * rollouts)
                cross = ((g1 * g2).sum(-1) - cross_noise.sum(-1)).mean()
                sig = [((e.mean(-1) ** 2).sum(-1) - (e.var(-1, ddof=1) / rollouts).sum(-1)).mean() for e in (e1, e2)]
                cell[f"fd_step_agreement_{key}"] = float(cross / math.sqrt(sig[0] * sig[1])) if min(sig) > 0 else float("nan")
        true_fresh = q[:, first_fresh:]
        cell["advantage"] = advantage_stats(true_fresh, critic_q[:, first_fresh:])
        within = cell["advantage"]["true_within_sd"] ** 2
        a0_column = layout["a0_probes"][0]
        cell["sampled"] = sampled_stats(q[:, a0_column], true_fresh, critic_q[:, a0_column], critic_baseline, target, within)
        adv = base.oracle_advantages(np.concatenate([q[:, a0_column:a0_column + 1], true_fresh], 1), 1)
        labels.update({f"a_true_a0_{h}": adv["a0"], f"a_true_a0_se_{h}": adv["a0_se"],
                       f"a_true_fresh_{h}": adv["fresh"], f"a_true_fresh_se_{h}": adv["fresh_se"],
                       f"q_mean_{h}": q.mean(-1), f"q_se_{h}": q.std(-1, ddof=1) / math.sqrt(rollouts)})
        results[f"{h}"] = cell
    return results, labels


def print_report(results, meta):
    print("Per horizon (pooled cos: raw / noise-corrected +- bootstrap SE / split-half; batch-mean direction cos raw "
          "(corrected); norm ratio |c|/|g_true|; oracle SE rms vs true-gradient rms per component):", flush=True)
    for h, cell in results.items():
        for key in [name[len("grad_"):] for name in cell if name.startswith("grad_")]:
            s = cell[f"grad_{key}"]
            print(f"  H={h:>3} grad@{key:9s} cos mean {s['cos_mean']:+.3f} | pooled {s['cos_pooled_raw']:+.3f} / "
                  f"{s['cos_pooled_corrected']:+.3f}+-{s['cos_pooled_corrected_se']:.3f} / {s['cos_pooled_splithalf']:+.3f} | "
                  f"batch {s['batch_cos_raw']:+.3f} ({s['batch_cos_corrected']:+.3f}) | norm {s['norm_ratio']:.2f} | "
                  f"SE {s['oracle_se_rms']:.3f} vs true {s['true_grad_rms']:.3f} (snr {s['oracle_snr']:.2f}, "
                  f"split-half self-cos {s['oracle_splithalf_cos_mean']:+.3f}, top1%/5% energy {s['energy_share_top1']:.2f}/"
                  f"{s['energy_share_top5']:.2f}) | critic FD cos {s['critic_fd_cos_mean']:+.3f} "
                  f"({s['critic_fd_cos_pooled_corrected']:+.3f}), autograd-vs-FD {s['critic_autograd_vs_fd_cos']:+.3f}",
                  flush=True)
        agreement = {key: value for key, value in cell.items() if key.startswith("fd_step_agreement")}
        if agreement:
            steps = meta["fd_steps"]
            print(f"  H={h:>3} true-gradient agreement between h={steps[0]:g} and h={steps[1]:g} (pooled, noise-corrected): "
                  + " ".join(f"{key[len('fd_step_agreement_'):]} {value:+.3f}" for key, value in agreement.items()), flush=True)
        a, sp = cell["advantage"], cell["sampled"]
        print(f"  H={h:>3} advantage (K={meta['K']}): corr {a['corr_raw']:+.3f} (corrected {a['corr_corrected']:+.3f}, "
              f"share {a['explained_share']:.3f}), slope true|critic {a['slope_true_on_critic']:.3f}, sd ratio "
              f"{a['sd_ratio_critic_over_true']:.2f}, true within sd {a['true_within_sd']:.4f}, noise share "
              f"{a['noise_share']:.3f}", flush=True)
        print(f"  H={h:>3} a0: A_true sd {sp['a_true_sd']:.4f} se {sp['a_true_se_rms']:.4f} baseline err "
              f"{sp['baseline_error_sd']:.4f} (rel {sp['reliability']:.3f}); lambda-return corr "
              f"{sp['lambda_return']['corr']:+.3f} ({sp['lambda_return']['corr_corrected']:+.3f}) sd x"
              f"{sp['lambda_return']['sd_ratio']:.2f}; critic corr {sp['critic']['corr']:+.3f} "
              f"({sp['critic']['corr_corrected']:+.3f}) sd x{sp['critic']['sd_ratio']:.2f}; value corr "
              f"{sp['value']['corr']:.3f} bias {sp['value']['bias']:+.3f} err sd {sp['value']['error_sd']:.3f} "
              f"(true sd {sp['value']['true_sd']:.3f})", flush=True)


@torch.no_grad()
def critic_baseline(oracle, obs, index, samples):
    """E_pi Q_critic(s, .) from `samples` draws of a stream disjoint from the oracle's K fresh actions."""
    keys = base.stream_key(oracle.seed_hash, BASELINE, torch.as_tensor(index, device=oracle.device)[:, None],
                           torch.arange(samples, device=oracle.device))
    actions, _ = oracle.sample(obs[:, None].expand(-1, samples, -1), keys)
    return oracle.critic_q(obs, actions).double().mean(1).cpu()


# ---------------------------------------------------------------- driver

def run_snapshot(path, cli, trainer, device, pool):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if snapshot["args"]["env_id"] != ENV_ID:
        raise ValueError(f"the oracle simulator is {ENV_ID}-only")
    physics = base.Physics(cli.threads, cli.seed)
    oracle = Oracle(trainer, snapshot, physics, cli, device)
    checkpoints = sorted(set(cli.checkpoints) | {cli.horizon})
    header = {"snapshot": str(path), "global_step": int(snapshot["global_step"]), "temperature": oracle.temperature,
              "n": cli.states, "M": cli.rollouts, "H": cli.horizon, "K": cli.fresh, "fd_steps": cli.fd_steps,
              "value_samples": cli.value_samples, "checkpoints": checkpoints}
    print(json.dumps(header), flush=True)
    validation = validate(oracle, trainer, cli)

    start = time.perf_counter()
    data = collect(oracle, trainer, int(oracle.args.num_envs), int(oracle.args.num_steps), cli.seed)
    rng = np.random.default_rng(cli.seed)
    index = np.sort(rng.choice(len(data["age"]), cli.states, replace=False))
    rows = torch.as_tensor(data["row"][index], device=device)
    obs = oracle.observe(torch.as_tensor(data["state"][index], device=device), rows)
    a0 = data["actions"][torch.as_tensor(index)].to(device)
    with torch.no_grad():
        mean, std = oracle.moments(obs)
        star = torch.tanh(mean)
        header["policy"] = {"pre_tanh_std_mean": float(std.mean()), "mean_abs_action_star": float(star.abs().mean()),
                            "action_spread_sd": float((torch.tanh(mean + std) - torch.tanh(mean - std)).mean() / 2)}
        print(f"policy at the oracle states: {header['policy']}", flush=True)
        fresh, _ = oracle.sample(obs[:, None].expand(-1, cli.fresh, -1),
                                 base.stream_key(oracle.seed_hash, FRESH, torch.as_tensor(index, device=device)[:, None],
                                                 torch.arange(cli.fresh, device=device)))
    centers = {"star": star, "a0": a0}
    blocks, spans = [], {}
    for key, center in centers.items():
        block, spans[key] = probes(center, cli.fd_steps)
        blocks.append(block)
    width = blocks[0].shape[1]  # 1 + 2 D len(fd_steps) per centre
    offsets = {key: i * width for i, key in enumerate(centers)}
    actions = torch.cat([*blocks, fresh], 1)  # (n, 2 width + K, D)
    first_fresh = 2 * width
    steps = cli.states * actions.shape[1] * cli.rollouts * cli.horizon
    print(f"collected chunk + {cli.states} states in {time.perf_counter() - start:.0f}s; {actions.shape[1]} actions x "
          f"{cli.rollouts} rollouts x {cli.horizon} steps = {steps / 1e6:.0f}M env steps, ~"
          f"{steps / validation['physics']['env_steps_per_s'] / 60:.1f} min", flush=True)

    start = time.perf_counter()
    returns = oracle_returns(oracle, cli, data, index, actions, pool)
    oracle_s = time.perf_counter() - start

    with torch.no_grad():
        critic_q = oracle.critic_q(obs, actions).double().cpu()
    to_t = lambda value: torch.as_tensor(np.asarray(value))  # noqa: E731
    dataset = {
        "meta": {**header, "gamma": oracle.gamma, "lambda": oracle.lam, "reward_scale": oracle.reward_scale,
                 "obs_clip": oracle.obs_clip, "fd_steps": list(cli.fd_steps),
                 "action_layout": {**{f"{key}_probes": [offset, offset + width] for key, offset in offsets.items()},
                                   "fresh": [first_fresh, actions.shape[1]]},
                 "probe_layout": "centre, then per fd step: centre + h e_d (d < D), centre - h e_d "
                                 "(the centre where a probe would leave [-1, 1])"},
        "norm": {"obs_means": to_t(snapshot["obs_means"]), "obs_variances": to_t(snapshot["obs_variances"]),
                 "obs_epsilon": float(snapshot["obs_epsilon"])},
        "states": {"index": to_t(index), "state": to_t(data["state"][index]), "age": to_t(data["age"][index]),
                   "row": to_t(data["row"][index]), "obs": obs.cpu(), "actions": actions.cpu(),
                   **{f"span_{key}": torch.stack(value).cpu() for key, value in spans.items()},
                   "critic_q": critic_q, "critic_baseline": critic_baseline(oracle, obs, index, cli.baseline_samples),
                   **{f"critic_grad_{key}": oracle.critic_grad(obs, center).cpu() for key, center in centers.items()}},
        "returns": {f"{h}": to_t(q) for h, q in returns.items()},  # (n, A, M) soft Q per rollout
        "chunk": {key: (to_t(value) if isinstance(value, np.ndarray) else value) for key, value in data.items()},
    }
    print(f"oracle {oracle_s:.0f}s.", flush=True)
    return finish(dataset, {"header": header, "validation": validation, "oracle_seconds": oracle_s},
                  cli.out_dir / f"{Path(path).parent.name}_{snapshot['global_step']}")


def finish(dataset, report, stem):
    """Score, print, and save the dataset (.pt, with labels over all states) and the report (.json)."""
    report["results"] = {}
    for name, mask in subsets(dataset).items():
        results, labels = score(dataset, mask)
        print(f"== {name} states ({int(mask.sum())} of {mask.size})", flush=True)
        print_report(results, dataset["meta"])
        report["results"][name] = results
        if name == "all":
            dataset["labels"] = {key: torch.as_tensor(np.asarray(value)) for key, value in labels.items()}
    dataset["states"].update({f"subset_{name}": torch.as_tensor(mask) for name, mask in subsets(dataset).items()})
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float))
    torch.save(dataset, stem.with_suffix(".pt"))
    print(f"saved {stem}.pt and .json", flush=True)
    return report


def rescore(path, cli, trainer, device):
    """Recompute the statistics of a saved dataset (adding the independent critic baseline if absent)."""
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    states = dataset["states"]
    if "critic_baseline" not in states:
        snapshot = torch.load(dataset["meta"]["snapshot"], map_location="cpu", weights_only=False)
        oracle = Oracle(trainer, snapshot, base.Physics(1, cli.seed), cli, device)
        states["critic_baseline"] = critic_baseline(oracle, states["obs"].to(device), states["index"].numpy(),
                                                    cli.baseline_samples)
    old = path.with_suffix(".json")
    report = json.loads(old.read_text()) if old.exists() else {}
    report.pop("results", None)
    return finish(dataset, report, path.with_suffix(""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--states", type=int, default=256, help="n states from the on-policy chunk")
    parser.add_argument("--rollouts", type=int, default=8, help="M CRN rollouts per (state, action)")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--fresh", type=int, default=8, help="K fresh actions per state")
    parser.add_argument("--value-samples", type=int, default=8, help="policy samples in the bootstrap V(s)")
    parser.add_argument("--fd-steps", type=float, nargs="+", default=[0.05],
                        help="finite-difference steps h in action space (each adds 4D actions per state)")
    parser.add_argument("--group-states", type=int, default=128, help="states per branch group")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-episodes", type=int, default=256, help="simulator episodes in the return check")
    parser.add_argument("--gym-episodes", type=int, default=64, help="gymnasium episodes in the return check")
    parser.add_argument("--baseline-samples", type=int, default=64,
                        help="independent critic draws in the estimates' E_pi Q_critic(s, .) baseline")
    parser.add_argument("--rescore", action="store_true", help="the paths are saved datasets: recompute statistics")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs" / "reppo_oracle")
    cli = parser.parse_args()
    if not all(1 <= h <= cli.horizon for h in cli.checkpoints) or cli.rollouts < 2 or cli.fresh < 2:
        raise ValueError("--checkpoints must lie in [1, --horizon]; --rollouts and --fresh need >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    trainer = load_trainer()
    trainer.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(cli.seed)
    device = torch.device("cuda")
    with ThreadPoolExecutor(max_workers=1) as pool:
        for path in cli.snapshots:
            if cli.rescore:
                rescore(path, cli, trainer, device)
            else:
                run_snapshot(path, cli, trainer, device, pool)


if __name__ == "__main__":
    main()
