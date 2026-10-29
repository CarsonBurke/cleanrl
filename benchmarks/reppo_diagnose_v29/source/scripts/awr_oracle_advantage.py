"""How far is one PPO update from the true policy gradient, and which noise source dominates?

    python scripts/awr_oracle_advantage.py SNAPSHOT.pt [SNAPSHOT.pt ...] [--out results.json]

MuJoCo physics is deterministic and its state (qpos, qvel, solver warmstart) exactly restorable, so
the true advantage A(s, a) = Q^pi(s, a) - V^pi(s) of the frozen snapshot policy can be measured by
branching from recorded states. For a HalfCheetah-v4 snapshot from scripts/awr_capture_policy.py:

1. Simulator. gymnasium's own HalfCheetah model, stepped by an embedded multi-threaded C kernel over
   flat (qpos, qvel, qacc_warmstart) state arrays; obs/reward/time-limit as HalfCheetah-v4; the
   snapshot's frozen observation and reward normalization (each state keeps its env's row); policy and
   critic batched on CUDA. Asserted: bitwise obs and reward (1e-6) against gymnasium on replayed action
   sequences, and the frozen policy's episode return against the trainer's own pipeline
   (awr_update_bench.collect).
2. Data. 16 envs x 1024-step chunks from staggered phases, GAE for every lambda exactly as the trainer
   computes it (truncation bootstraps). TRAIN = the first --train-chunks chunks (each one real batch);
   EVAL = a random subset of states from --eval-chunks later chunks (the full chunks score the bench's
   usual held-out dJ). --train-chunks 0 skips the TRAIN oracle (anatomy then runs on EVAL);
   --model-chunks adds later chunks that --save-data stores with the EVAL oracle for
   scripts/awr_advantage_model_bench.py.
3. Oracle Q. Per state: the sampled action a_0 and K fresh actions a_k ~ pi_old; per action M
   rollouts of H steps under the frozen stochastic policy, bootstrapped with gamma^H V(s_H) (or V at
   the 1000-step truncation, as the trainer does), with partial returns at every --checkpoints horizon.
   Rollout m of every action at a state uses the same future base randomness (common random numbers):
   Beta = X / (X + Y), X, Y ~ Gamma by Marsaglia-Tsang on a fixed candidate budget whose normal and
   uniform draws come from a counter-based hash of (state, m, t, dim); its marginals are KS-tested.
   A(s, a_0) = Q(a_0) - mean_k Q(a_k); for fresh actions the leave-one-out baseline,
   A(s, a_k) = K / (K - 1) (Q(a_k) - mean_k Q), keeps the estimate unbiased. SEs come from the m-spread,
   paired across actions.
4. Anatomy of A_hat_lambda against A_true on the sampled action: correlation, the share of Var(A_hat)
   that A_true explains (raw and oracle-noise corrected), the within-state true-advantage variance over
   Var(A_hat), and the per-state offset in A_hat - A_true (env-half cross-fitted ridge on state features,
   plus its correlation with the oracle's critic error V^pi(s) - V(s)).
5. Update ceiling. awr_update_bench's real update (snapshot Adam geometry, 10 full-batch epochs, nGPT
   normalization, clip, lr sweep) for PPO on A_hat_0.95 / 0.8 / 0.5, oracle-1 (PPO on A_true(s, a_0))
   and oracle-K (PPO clipped surrogate over the K fresh actions per state with A_true: the per-state
   policy gradient without one-action score noise), the oracle forces rescaled to A_hat_0.95's std.
   Each point is scored on EVAL by the bench's dJ (A_hat_0.95 / MC) and by the oracle
   dJ = mean_{s,k} (r(a_k | s) - 1) A_true(s, a_k), with exact Beta KL, its location/concentration
   split and the entropy change; the report gives each rule's oracle dJ at matched KL (--skip-ceiling).

Run it through mlq (CUDA). Physics dominates: ~(n_train (K + 1) + n_eval K) M H env steps.
"""

import argparse
import copy
import ctypes
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awr_update_bench as bench  # noqa: E402  (also puts the repo root on sys.path)

from cleanrl.shared.ppo_loop import compute_gae  # noqa: E402
from cleanrl.shared.staggered_envs import compute_phase_offsets  # noqa: E402

ENV_ID = "HalfCheetah-v4"
FRAME_SKIP = 5
TIME_LIMIT = 1000
FORWARD_REWARD_WEIGHT, CTRL_COST_WEIGHT = 1.0, 0.1
NORM_EPSILON, NORM_CLIP = 1e-8, 10.0  # VectorObsNorm / VectorRewardNorm defaults
SAMPLE_EPS = 1e-6  # the trainer's native-action clamp
LAMBDAS = bench.TARGET_LAMBDAS
# Counter-based random streams: every draw is a hash of (stream, ...), so any branch can recreate it.
COLLECT, FRESH, BRANCH, TEST, VALIDATE = 1, 2, 3, 4, 5


# ---------------------------------------------------------------- physics

KERNEL_SOURCE = r"""
/* Branch stepping over flat states: each row is (qpos, qvel, qacc_warmstart), which with ctrl is
 * everything mj_step reads for a model without actuator activations, mocap or applied forces
 * (HalfCheetah). Rows are restored into one scratch mjData per thread, stepped frame_skip times and
 * written back, so any number of branches share a few KB of state each. Threads are created per
 * call: a call steps tens of thousands of rows, so their start-up cost is noise. */
#include <mujoco/mujoco.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

enum { BLOCK = 16 };

typedef struct {
  const mjModel *model;
  double *state;
  const float *ctrl;
  int count, frame_skip;
  _Atomic int cursor;
} job_t;

typedef struct {
  job_t *job;
  mjData *data;
} worker_t;

static void *work(void *argument) {
  worker_t *worker = (worker_t *)argument;
  job_t *job = worker->job;
  const mjModel *m = job->model;
  mjData *d = worker->data;
  const int nq = m->nq, nv = m->nv, nu = m->nu, width = nq + 2 * nv;
  for (;;) {
    const int start = atomic_fetch_add_explicit(&job->cursor, BLOCK, memory_order_relaxed);
    if (start >= job->count) {
      return NULL;
    }
    const int stop = start + BLOCK < job->count ? start + BLOCK : job->count;
    for (int i = start; i < stop; ++i) {
      double *row = job->state + (size_t)i * width;
      const float *action = job->ctrl + (size_t)i * nu;
      memcpy(d->qpos, row, (size_t)nq * sizeof(mjtNum));
      memcpy(d->qvel, row + nq, (size_t)nv * sizeof(mjtNum));
      memcpy(d->qacc_warmstart, row + nq + nv, (size_t)nv * sizeof(mjtNum));
      for (int j = 0; j < nu; ++j) {
        d->ctrl[j] = (mjtNum)action[j];  /* gymnasium assigns the float32 action to float64 ctrl */
      }
      for (int frame = 0; frame < job->frame_skip; ++frame) {
        mj_step(m, d);
      }
      memcpy(row, d->qpos, (size_t)nq * sizeof(mjtNum));
      memcpy(row + nq, d->qvel, (size_t)nv * sizeof(mjtNum));
      memcpy(row + nq + nv, d->qacc_warmstart, (size_t)nv * sizeof(mjtNum));
    }
  }
}

/* Returns the number of threads that ran (the caller is one of them). */
int oracle_step(const mjModel *model, mjData **scratch, int threads, double *state, const float *ctrl,
                int count, int frame_skip) {
  job_t job = {model, state, ctrl, count, frame_skip};
  atomic_init(&job.cursor, 0);
  worker_t workers[threads];
  pthread_t handles[threads];
  int started = 1;
  for (int i = 1; i < threads; ++i, ++started) {
    workers[i] = (worker_t){&job, scratch[i]};
    if (pthread_create(&handles[i], NULL, work, &workers[i]) != 0) {
      break;
    }
  }
  workers[0] = (worker_t){&job, scratch[0]};
  work(&workers[0]);
  for (int i = 1; i < started; ++i) {
    pthread_join(handles[i], NULL);
  }
  return started;
}
"""


def build_kernel():
    """Compile KERNEL_SOURCE once per (source, MuJoCo build) into the user cache, like mujoco_env."""
    package = Path(mujoco.__file__).resolve().parent
    libraries = sorted(package.glob("libmujoco.so.*"))
    if len(libraries) != 1:
        raise RuntimeError("the oracle kernel needs exactly one installed libmujoco.so")
    library = libraries[0]
    fingerprint = hashlib.sha256(KERNEL_SOURCE.encode() + str(library).encode()
                                 + mujoco.__version__.encode()).hexdigest()[:20]
    directory = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cleanrl" / "oracle-native" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "oracle_kernel.so"
    if not output.exists():
        source = directory / "oracle_kernel.c"
        source.write_text(KERNEL_SOURCE)
        fd, temporary = tempfile.mkstemp(suffix=".so", dir=directory)
        os.close(fd)
        try:
            # No fast-math: MuJoCo's arithmetic must stay bit-identical to gymnasium's.
            subprocess.run(["cc", "-O2", "-std=c11", "-fPIC", "-shared", "-pthread", "-I", str(package / "include"),
                            str(source), str(library), f"-Wl,-rpath,{package}", "-o", temporary],
                           check=True, capture_output=True, text=True)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(output))
    lib.oracle_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                                np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]
    lib.oracle_step.restype = ctypes.c_int
    return lib


class Physics:
    """gymnasium's own HalfCheetah-v4 model and reset, stepped in place over (n, nq + 2 nv) state rows."""

    def __init__(self, threads, seed):
        # The unwrapped env supplies the exact model (XML asset, options) and reset distribution.
        self.env = gym.make(ENV_ID).unwrapped
        self.env.reset(seed=seed)
        self.model, self.data = self.env.model, self.env.data
        if self.env.frame_skip != FRAME_SKIP or self.model.na != 0:
            raise ValueError("unexpected HalfCheetah-v4 configuration")
        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu
        self.width = self.nq + 2 * self.nv
        self.dt = self.env.dt
        self.threads = threads
        self.scratch = [mujoco.MjData(self.model) for _ in range(threads)]
        self._scratch = (ctypes.c_void_p * threads)(*(data._address for data in self.scratch))
        self._lib = build_kernel()

    def read(self, data=None):
        data = self.data if data is None else data
        return np.concatenate((data.qpos, data.qvel, data.qacc_warmstart))

    def reset(self, count):
        """`count` fresh episode starts from gymnasium's own reset (its seeded np_random)."""
        rows = np.empty((count, self.width))
        for row in rows:
            self.env.reset()
            row[:] = self.read()
        return rows

    def step(self, state, ctrl):
        if state.shape[0] != ctrl.shape[0] or state.shape[1] != self.width or ctrl.shape[1] != self.nu:
            raise ValueError("state/ctrl shape mismatch")
        self._lib.oracle_step(self.model._address, self._scratch, self.threads, state, ctrl,
                              state.shape[0], FRAME_SKIP)


# ---------------------------------------------------------------- counter-based coupled Beta sampler
# 32-bit hashing in int64 tensors: every product is split so it stays below 2^63.

MASK = 0xFFFFFFFF


def mul32(x, constant):
    return (x * (constant & 0xFFFF) + (((x * (constant >> 16)) & 0xFFFF) << 16)) & MASK


def mix32(x):
    """lowbias32 (Wellons): a well-avalanched bijection of 32-bit words."""
    x = x ^ (x >> 16)
    x = mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def fold(h, part):
    """Injective in `part` for fixed h: the key of a child stream."""
    return mix32(h ^ mul32(part & MASK, 0x9E3779B9))


def stream_key(*parts):
    h = parts[0]
    for part in parts[1:]:
        h = fold(h, part)
    return h


def uniforms(base, count):
    """(..., count) uniforms in (0, 1) at 24-bit resolution, draw j = hash(base, j)."""
    bits = mix32(fold(base[..., None], torch.arange(count, device=base.device)))
    return ((bits >> 8).to(torch.float32) + 0.5) * 2.0 ** -24


def gamma_mt(shape, normal_u, accept_u, boost_u):
    """Marsaglia-Tsang Gamma(shape) with a fixed candidate budget (first accepted wins).

    Coupling needs every branch to consume the same draws, so there is no loop: all candidates are
    evaluated and the first accepted one is taken. Acceptance is >= 95% per candidate for shape >= 1,
    so with 5 candidates the all-rejected fallback (the proposal's centre d) has probability < 1e-6.
    shape < 1 uses Gamma(shape + 1) U^(1/shape)."""
    boosted = shape < 1.0
    a = torch.where(boosted, shape + 1.0, shape)
    d = a - 1.0 / 3.0
    c = torch.rsqrt(9.0 * d)
    x = math.sqrt(2.0) * torch.erfinv(2.0 * normal_u - 1.0)
    v = (1.0 + c[..., None] * x) ** 3
    ok = (v > 0) & (torch.log(accept_u) < 0.5 * x * x + d[..., None] * (1.0 - v + torch.log(v)))
    sample = d
    for j in reversed(range(x.shape[-1])):
        sample = torch.where(ok[..., j], d * v[..., j], sample)
    return torch.where(boosted, sample * boost_u ** (1.0 / shape), sample)


def coupled_beta(alpha, beta, base, candidates):
    """Beta(alpha, beta) native actions from the base keys' draws, clamped as the trainer does."""
    draws = uniforms(base, 4 * candidates + 2)
    c = candidates
    x = gamma_mt(alpha, draws[..., :c], draws[..., 2 * c:3 * c], draws[..., 4 * c])
    y = gamma_mt(beta, draws[..., c:2 * c], draws[..., 3 * c:4 * c], draws[..., 4 * c + 1])
    return (x / (x + y)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)


def validate_sampler(device, seed_hash, candidates):
    """Assert the hash matches a uint32 reference and the sampler's marginals are Beta (KS)."""
    words = np.random.default_rng(0).integers(0, 2 ** 32, 100_000, dtype=np.uint64)
    reference = words.astype(np.uint32)
    with np.errstate(over="ignore"):
        reference ^= reference >> np.uint32(16)
        reference *= np.uint32(0x7FEB352D)
        reference ^= reference >> np.uint32(15)
        reference *= np.uint32(0x846CA68B)
        reference ^= reference >> np.uint32(16)
    ours = mix32(torch.as_tensor(words.astype(np.int64), device=device)).cpu().numpy()
    assert np.array_equal(ours, reference.astype(np.int64)), "mix32 disagrees with the uint32 reference"
    count = 200_000
    base = stream_key(seed_hash, TEST, torch.arange(count, device=device))
    report = {}
    for a, b in ((1.0, 1.0), (1.3, 4.0), (3.0, 3.0), (20.0, 7.0), (250.0, 90.0), (0.4, 0.7)):
        alpha = torch.full((count,), a, device=device)
        beta = torch.full((count,), b, device=device)
        sample = coupled_beta(alpha, beta, base, candidates).double().cpu().numpy()
        # The trainer's clamp puts F(eps) + 1 - F(1 - eps) on the bounds (0.3% for shape 0.4): test the
        # interior against the conditional CDF and the clamped mass against its binomial SE.
        law = stats.beta(a, b)
        edge = np.float64(np.float32(SAMPLE_EPS)), np.float64(np.float32(1.0 - SAMPLE_EPS))
        low_mass, high_mass = law.cdf(edge[0]), law.sf(edge[1])
        interior = sample[(sample > edge[0]) & (sample < edge[1])]
        ks = stats.kstest(interior, lambda x: (law.cdf(x) - low_mass) / (1.0 - low_mass - high_mass))
        expected = (low_mass + high_mass) * count
        assert abs(count - interior.size - expected) <= 6 * math.sqrt(expected) + 3, f"clamped mass off for ({a}, {b})"
        torch_sample = torch.distributions.Beta(alpha.double(), beta.double()).sample().cpu().numpy()
        report[f"{a:g},{b:g}"] = {"ks": float(ks.statistic), "p": float(ks.pvalue),
                                  "mean": float(sample.mean()), "torch_mean": float(torch_sample.mean()),
                                  "var": float(sample.var()), "torch_var": float(torch_sample.var())}
        assert ks.pvalue > 1e-4, f"coupled Beta({a}, {b}) fails KS: {ks}"
        assert abs(sample.mean() - torch_sample.mean()) < 6 * math.sqrt(2 * torch_sample.var() / count)
    # Coupling: a small parameter change moves the same draw only slightly.
    alpha = torch.full((count,), 5.0, device=device)
    near = coupled_beta(alpha, alpha, base, candidates)
    moved = coupled_beta(alpha * 1.05, alpha, base, candidates)
    report["coupling_corr_5pct"] = float(torch.corrcoef(torch.stack((near, moved)))[0, 1])
    return report


# ---------------------------------------------------------------- frozen policy

class Oracle:
    """Frozen snapshot policy + critic on CUDA, with the snapshot's normalization statistics frozen."""

    def __init__(self, agent, snapshot, physics, cli, device):
        self.agent, self.physics, self.device = agent, physics, device
        self.gamma = float(snapshot["args"]["gamma"])
        self.candidates, self.rollouts, self.horizon = cli.candidates, cli.rollouts, cli.horizon
        self.checkpoints = sorted(set(cli.checkpoints) | {cli.horizon})
        self.seed_hash = int(mix32(cli.seed + 0x5EED))
        as64 = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)  # noqa: E731
        self.obs_mean = as64(snapshot["obs_norm"]["means"])
        self.obs_scale = torch.sqrt(as64(snapshot["obs_norm"]["variances"]) + NORM_EPSILON)
        self.reward_scale = torch.sqrt(as64(snapshot["rew_norm"]["variances"]) + NORM_EPSILON)
        self.low, self.span = agent.action_low, agent.action_high - agent.action_low
        self.dims = torch.arange(agent.action_dim, device=device)
        # One small tensor per step (not views of one buffer), so the compiled step never re-guards.
        self.discounts = [torch.tensor([self.gamma ** max(t - 1, 0), self.gamma ** t], dtype=torch.float64,
                                       device=device) for t in range(self.horizon + 1)]
        self.steps = [torch.tensor(t, device=device) for t in range(self.horizon + 1)]
        self._step_fn = torch.compile(self._branch_step, fullgraph=True, dynamic=False)

    def observe(self, state, rows):
        """HalfCheetah obs (qpos[1:], qvel), normalized with frozen per-env statistics, as VectorObsNorm."""
        obs = state[:, 1:1 + self.physics.nq - 1 + self.physics.nv]
        return ((obs - self.obs_mean[rows]) / self.obs_scale[rows]).clamp(-NORM_CLIP, NORM_CLIP).float()

    def raw_reward(self, x_after, x_before, action):
        forward = FORWARD_REWARD_WEIGHT * ((x_after - x_before) / self.physics.dt)
        return forward - CTRL_COST_WEIGHT * action.double().square().sum(-1)

    def normalize_reward(self, raw, rows):
        return (raw / self.reward_scale[rows]).clamp(-NORM_CLIP, NORM_CLIP).float()

    def physical(self, native):
        return self.low + self.span * native

    def sample(self, alpha, beta, key):
        return coupled_beta(alpha, beta, fold(key[..., None], self.dims), self.candidates)

    def _branch_step(self, state, x_before, action, total, alive, ages, rows, sids, rollout, t, discount):
        """Consume physics step t - 1 (taken with `action`), return the step-t action.

        `t` and `discount` = (gamma^(t-1), gamma^t) are tensors so one graph serves every step."""
        raw = self.raw_reward(state[:, 0], x_before, action)
        total = total + torch.where(alive, discount[0] * self.normalize_reward(raw, rows).double(), 0.0)
        obs = self.observe(state, rows)
        alpha, beta, value = self.agent.get_policy_and_value(obs)
        value = value.flatten().double()
        # The trainer's truncation bootstrap: the time limit ends the trace at V(final obs).
        truncated = alive & (ages + t >= TIME_LIMIT)
        total = total + torch.where(truncated, discount[1] * value, 0.0)
        alive = alive & ~truncated
        native = self.sample(alpha, beta, stream_key(self.seed_hash, BRANCH, sids, rollout, t))
        # x_before is cloned: state is a buffer the next step overwrites.
        return total, alive, value, self.physical(native), state[:, 0].clone()


class BranchGroup:
    """All (state, action, rollout) branches of one group of states, stepped in lockstep."""

    def __init__(self, oracle, states, ages, rows, sids, natives):
        count, actions, _ = natives.shape
        device, per = oracle.device, actions * oracle.rollouts
        self.oracle, self.shape = oracle, (count, actions, oracle.rollouts)
        size = count * per
        self.state = torch.empty((size, oracle.physics.width), dtype=torch.float64).pin_memory()
        self.state_np = self.state.numpy()
        self.state_np[:] = np.repeat(states, per, axis=0)
        self.state_gpu = torch.empty(self.state.shape, dtype=torch.float64, device=device)
        self.ctrl = torch.empty((size, oracle.agent.action_dim), dtype=torch.float32).pin_memory()
        self.ctrl_np = self.ctrl.numpy()
        expand = lambda values: torch.as_tensor(values, device=device).repeat_interleave(per)  # noqa: E731
        self.ages, self.rows, self.sids = expand(ages), expand(rows), expand(sids)
        self.rollout = torch.arange(oracle.rollouts, device=device).repeat(count * actions)
        self.action = oracle.physical(natives.reshape(-1, natives.shape[-1])).repeat_interleave(oracle.rollouts, 0)
        self.ctrl.copy_(self.action)
        self.x_before = expand(states[:, 0])
        self.total = torch.zeros(size, dtype=torch.float64, device=device)
        self.alive = torch.ones(size, dtype=torch.bool, device=device)
        self.returns = {}

    def advance(self, t):
        oracle = self.oracle
        self.state_gpu.copy_(self.state, non_blocking=True)
        self.total, self.alive, value, self.action, self.x_before = oracle._step_fn(
            self.state_gpu, self.x_before, self.action, self.total, self.alive, self.ages, self.rows,
            self.sids, self.rollout, oracle.steps[t], oracle.discounts[t])
        if t in oracle.checkpoints:
            bootstrap = torch.where(self.alive, oracle.discounts[t][1] * value, 0.0)
            self.returns[t] = (self.total + bootstrap).view(self.shape).cpu()
        if t < oracle.horizon:
            self.ctrl.copy_(self.action, non_blocking=True)
        torch.cuda.current_stream().synchronize()


@torch.no_grad()
def oracle_returns(oracle, states, ages, rows, sids, natives, group_states, pool, label):
    """Q estimates {H: (n, A, M) float64} for the given states and native actions (n, A, dim).

    Two groups are in flight: one group's GPU step overlaps the other's physics (ctypes drops the GIL).
    Groups are padded to a fixed size so one compiled graph serves them all."""
    count = len(states)
    padded = -(-count // group_states) * group_states
    pad = lambda values: np.concatenate([values, np.repeat(values[-1:], padded - count, 0)])  # noqa: E731
    states, ages, rows, sids = map(pad, (states, ages, rows, sids))
    natives = torch.cat([natives, natives[-1:].expand(padded - count, *natives.shape[1:])])
    starts = list(range(0, padded, group_states))
    out = {h: [] for h in oracle.checkpoints}
    begin = time.perf_counter()
    for index in range(0, len(starts), 2):
        groups = [BranchGroup(oracle, states[s:s + group_states], ages[s:s + group_states],
                              rows[s:s + group_states], sids[s:s + group_states], natives[s:s + group_states])
                  for s in starts[index:index + 2]]
        pending, owner = None, None
        for t in range(oracle.horizon + 1):
            for group in groups:
                if owner is group:
                    pending.result()
                    pending = owner = None
                if t > 0:
                    group.advance(t)
                if t < oracle.horizon:
                    if pending is not None:
                        pending.result()
                    pending, owner = pool.submit(oracle.physics.step, group.state_np, group.ctrl_np), group
        if pending is not None:
            pending.result()
        for group in groups:
            for h in oracle.checkpoints:
                out[h].append(group.returns[h])
        done = min(padded, starts[min(index + 2, len(starts)) - 1] + group_states)
        elapsed = time.perf_counter() - begin
        steps = done * natives.shape[1] * oracle.rollouts * oracle.horizon
        print(f"  oracle {label}: {done}/{padded} states, {elapsed:.0f}s, {steps / elapsed / 1e6:.2f}M env steps/s, "
              f"eta {elapsed * (padded - done) / done:.0f}s", flush=True)
    return {h: torch.cat(values)[:count].numpy() for h, values in out.items()}


# ---------------------------------------------------------------- validation

def validate_physics(oracle, seed):
    """Assert bitwise obs, reward (1e-6) and truncation parity with gymnasium, through the kernel's
    restore path: many interleaved copies of two trajectories are stepped by all threads each step."""
    physics = oracle.physics
    envs = [gym.make(ENV_ID) for _ in range(2)]
    rng = np.random.default_rng(seed)
    copies = 512
    state = np.empty((2 * copies, physics.width))
    for index, env in enumerate(envs):
        obs, _ = env.reset(seed=seed + index)
        row = physics.read(env.unwrapped.data)
        assert np.array_equal(row[1:physics.nq + physics.nv], obs), "reset state does not reproduce obs"
        state[index::2] = row
    worst_reward = 0.0
    for step in range(TIME_LIMIT):
        actions = rng.uniform(-1.0, 1.0, (2, physics.nu)).astype(np.float32)
        before = state[:, 0].copy()
        physics.step(state, np.ascontiguousarray(np.tile(actions, (copies, 1))))
        reward = oracle.raw_reward(torch.from_numpy(state[:, 0]), torch.from_numpy(before),
                                   torch.from_numpy(np.tile(actions, (copies, 1)))).numpy()
        for index, env in enumerate(envs):
            obs, gym_reward, terminated, truncated, _ = env.step(actions[index])
            mine = state[index::2, 1:physics.nq + physics.nv]
            assert (mine == obs).all(), f"obs diverged from gymnasium at step {step + 1}"
            worst_reward = max(worst_reward, float(np.abs(reward[index::2] - gym_reward).max()))
            assert not terminated and truncated == (step + 1 >= TIME_LIMIT), "time-limit mismatch"
    assert worst_reward < 1e-6, f"reward mismatch {worst_reward}"
    # Throughput for the ETA: one full-size batch of branch steps.
    batch = np.repeat(state[:2], 16384, axis=0)
    ctrl = rng.uniform(-1, 1, (batch.shape[0], physics.nu)).astype(np.float32)
    start = time.perf_counter()
    for _ in range(5):
        physics.step(batch, ctrl)
    rate = 5 * batch.shape[0] / (time.perf_counter() - start)
    return {"replayed_steps": TIME_LIMIT, "copies": 2 * copies, "max_reward_error": worst_reward,
            "obs_bitwise": True, "env_steps_per_s": rate}


@torch.no_grad()
def policy_episode_returns(oracle, episodes):
    """Raw returns of full episodes of the frozen policy in this simulator, from gymnasium resets."""
    physics, agent, device = oracle.physics, oracle.agent, oracle.device
    state = physics.reset(episodes)
    rows = torch.arange(episodes, device=device) % oracle.obs_mean.shape[0]
    lanes = torch.arange(episodes, device=device)
    returns = torch.zeros(episodes, dtype=torch.float64, device=device)
    for step in range(TIME_LIMIT):
        before = torch.from_numpy(state[:, 0]).to(device)
        alpha, beta = agent.policy(oracle.observe(torch.from_numpy(state).to(device), rows))
        action = oracle.physical(oracle.sample(alpha, beta, stream_key(oracle.seed_hash, VALIDATE, lanes, step)))
        physics.step(state, action.cpu().numpy())
        returns += oracle.raw_reward(torch.from_numpy(state[:, 0]).to(device), before, action)
    return returns.cpu().numpy()


# ---------------------------------------------------------------- data

@torch.no_grad()
def collect(oracle, chunks, num_envs, num_steps, seed):
    """Frozen-policy chunks from staggered phases, GAE per chunk exactly as the trainer computes it.

    Returns per chunk a dict of flattened (step, env) GPU tensors plus host states/ages/rows/sids."""
    physics, agent, device = oracle.physics, oracle.agent, oracle.device
    state = physics.reset(num_envs)
    age = np.zeros(num_envs, dtype=np.int64)
    rows = torch.arange(num_envs, device=device)
    clock = 0

    def act():
        nonlocal clock
        current = torch.from_numpy(state).to(device)
        obs = oracle.observe(current, rows)
        alpha, beta, value = agent.get_policy_and_value(obs)
        native = oracle.sample(alpha, beta, stream_key(oracle.seed_hash, COLLECT, rows, clock))
        clock += 1
        return current, obs, alpha, beta, value.flatten(), native

    offsets = compute_phase_offsets(num_envs, TIME_LIMIT, seed)
    for step in range(int(offsets.max())):
        native = act()[-1]
        stepped = state.copy()
        physics.step(stepped, oracle.physical(native).cpu().numpy())
        active = offsets > step
        state[active], age[active] = stepped[active], age[active] + 1
    batches = []
    for chunk in range(chunks):
        fields = {key: [] for key in ("obs", "native", "alpha", "beta", "value")}
        states = np.empty((num_steps, num_envs, physics.width))
        ages = np.empty((num_steps, num_envs), dtype=np.int64)
        rewards = torch.zeros(num_steps, num_envs, device=device)
        truncations = torch.zeros(num_steps, num_envs, device=device)
        truncation_values = torch.zeros(num_steps, num_envs, device=device)
        for step in range(num_steps):
            current, obs, alpha, beta, value, native = act()
            action = oracle.physical(native)
            states[step], ages[step] = state, age
            physics.step(state, action.cpu().numpy())
            after = torch.from_numpy(state).to(device)
            rewards[step] = oracle.normalize_reward(oracle.raw_reward(after[:, 0], current[:, 0], action), rows)
            age += 1
            done = np.flatnonzero(age >= TIME_LIMIT)
            if done.size:
                index = torch.as_tensor(done, device=device)
                truncations[step, index] = 1.0
                truncation_values[step, index] = agent.get_value(oracle.observe(after[index], rows[index])).flatten()
                state[done], age[done] = physics.reset(done.size), 0
            for key, value_ in zip(fields, (obs, native, alpha, beta, value)):
                fields[key].append(value_)
        tail = agent.get_value(oracle.observe(torch.from_numpy(state).to(device), rows)).flatten()
        values = torch.stack(fields["value"])
        entry = {key: torch.cat(value_) for key, value_ in fields.items() if key != "value"}
        entry["value"] = values.flatten()
        entry["logprob"] = agent.action_logprob(entry["alpha"], entry["beta"], entry["native"])
        terminations = torch.zeros_like(rewards)
        for lam in LAMBDAS:
            entry[f"advantages_{lam:g}"] = compute_gae(rewards, values, terminations, truncations, truncation_values,
                                                       tail, oracle.gamma, lam)[0].flatten()
        entry["advantages"], entry["advantages_mc"] = entry["advantages_0.95"], entry["advantages_1"]
        entry["trace"] = {"rewards": rewards, "values": values, "truncations": truncations,
                          "truncation_values": truncation_values, "tail": tail}
        entry["host"] = {"state": states.reshape(-1, physics.width), "age": ages.flatten(),
                         "row": np.tile(np.arange(num_envs), num_steps),
                         "sid": chunk * num_steps * num_envs + np.arange(num_steps * num_envs)}
        batches.append(entry)
    return batches


def select(batch, index):
    out = {key: value[torch.as_tensor(index, device=value.device)] for key, value in batch.items()
           if key not in ("host", "trace")}
    out["host"] = {key: value[index] for key, value in batch["host"].items()}
    return out


def merge(batches):
    out = bench.concatenate([{key: value for key, value in batch.items() if key not in ("host", "trace")}
                             for batch in batches])
    out["host"] = {key: np.concatenate([batch["host"][key] for batch in batches]) for key in batches[0]["host"]}
    return out


@torch.no_grad()
def fresh_actions(oracle, batch, count, chunk=4096):
    """K fresh native actions per state from pi_old and their old log-probs: (n, K, dim), (n, K)."""
    natives, logprobs = [], []
    sids = torch.as_tensor(batch["host"]["sid"], device=oracle.device)
    ks = torch.arange(count, device=oracle.device)
    for start in range(0, sids.shape[0], chunk):
        alpha = batch["alpha"][start:start + chunk, None].expand(-1, count, -1)
        beta = batch["beta"][start:start + chunk, None].expand(-1, count, -1)
        native = oracle.sample(alpha, beta, stream_key(oracle.seed_hash, FRESH, sids[start:start + chunk, None], ks))
        natives.append(native)
        logprobs.append(oracle.agent.action_logprob(alpha, beta, native))
    return torch.cat(natives), torch.cat(logprobs)


# ---------------------------------------------------------------- oracle advantages

def oracle_advantages(returns, fresh_from):
    """A_true and its m-spread SE from Q returns (n, A, M); actions fresh_from.. are the K fresh ones.

    Returns the sampled action's (A, SE) when fresh_from == 1, the fresh actions' leave-one-out
    (A, SE), V^pi(s) and the within-state true-advantage variance with rollout noise removed."""
    fresh = returns[:, fresh_from:]
    count = fresh.shape[1]
    rollouts = fresh.shape[2]
    baseline = fresh.mean(1)  # (n, M): V^pi(s) per rollout index, paired across actions by CRN
    out = {"v": baseline.mean(1)}
    per = count / (count - 1) * (fresh - baseline[:, None])
    out["fresh"], out["fresh_se"] = per.mean(2), per.std(2, ddof=1) / math.sqrt(rollouts)
    if fresh_from == 1:
        sampled = returns[:, 0] - baseline
        out["a0"], out["a0_se"] = sampled.mean(1), sampled.std(1, ddof=1) / math.sqrt(rollouts)
        # CRN gain: rollout variance of Q(a_0) alone over that of the paired difference.
        out["crn_gain"] = float(returns[:, 0].var(1, ddof=1).mean() / sampled.var(1, ddof=1).mean())
    centered = fresh - baseline[:, None]
    signal = centered.mean(2).var(1, ddof=1) - centered.var(2, ddof=1).sum(1) / (rollouts * (count - 1))
    out["within_var"] = float(signal.mean())
    out["within_var_raw"] = float(centered.mean(2).var(1, ddof=1).mean())
    return out


def precision(advantages):
    """Oracle noise against A_true's spread, for the sampled and the fresh actions."""
    out = {}
    for key in ("a0", "fresh"):
        if key in advantages:
            value, se = advantages[key], advantages[f"{key}_se"]
            out[key] = {"sd": float(value.std()), "se_rms": float(np.sqrt(np.mean(se ** 2))),
                        "noise_share": float(np.mean(se ** 2) / value.var())}
    out["within_sd"] = float(np.sqrt(max(advantages["within_var"], 0.0)))
    if "crn_gain" in advantages:
        out["crn_gain"] = advantages["crn_gain"]
    return out


def trunk_features(mlp, x):
    """A ResidualMLP's last hidden representation (the head's input direction)."""
    h = F.normalize(mlp.first(x), p=2, dim=-1)
    branch = F.normalize(mlp.second(8.0 * h), p=2, dim=-1)
    return F.normalize(h + branch, p=2, dim=-1)


RIDGE_GRID = np.logspace(-3, 3, 13)


def ridge_fit(x, y, penalty):
    mean, scale = x.mean(0), x.std(0) + 1e-8
    z = (x - mean) / scale
    weights = np.linalg.solve(z.T @ z + penalty * len(z) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return lambda new: ((new - mean) / scale) @ weights + y.mean()


def crossfit_ridge(x, y, folds, inner):
    """Out-of-fold ridge predictions: fit on one env half (penalty chosen on its own env halves)."""
    prediction = np.empty_like(y)
    for held in (False, True):
        train = folds != held
        xt, yt, it = x[train], y[train], inner[train]
        losses = [sum(np.mean((ridge_fit(xt[it != side], yt[it != side], p)(xt[it == side]) - yt[it == side]) ** 2)
                      for side in (False, True)) for p in RIDGE_GRID]
        prediction[~train] = ridge_fit(xt, yt, RIDGE_GRID[int(np.argmin(losses))])(x[~train])
    return prediction


def anatomy(train, advantages_by_h, primary, features):
    """A_hat_lambda against A_true(s, a_0) on the TRAIN states."""
    oracle = advantages_by_h[primary]
    truth, se = oracle["a0"], oracle["a0_se"]
    reliability = 1.0 - np.mean(se ** 2) / truth.var()
    critic_error = oracle["v"] - train["value"].double().cpu().numpy()  # V^pi(s) - V(s)
    rows = train["host"]["row"]
    folds, inner = rows >= rows.max() // 2 + 1, rows % 2 == 1
    table = {}
    for lam in LAMBDAS:
        estimate = train[f"advantages_{lam:g}"].double().cpu().numpy()
        variance = estimate.var()
        corr = float(np.corrcoef(estimate, truth)[0, 1])
        error = estimate - truth
        offset = crossfit_ridge(features, error, folds, inner)
        explained = error.var() - (error - offset).var()
        table[f"{lam:g}"] = {
            "corr": corr, "r2": corr ** 2, "r2_signal": corr ** 2 / reliability,
            "slope_true_on_hat": float(np.cov(truth, estimate)[0, 1] / variance),
            "within_true_over_var_hat": oracle["within_var"] / variance,
            "error_var_share": float(error.var() / variance),
            "offset_share": float(explained / variance),
            "offset_r2_of_error": float(explained / error.var()),
            "corr_error_critic": float(np.corrcoef(error, critic_error)[0, 1]),
            "slope_error_on_critic": float(np.cov(error, critic_error)[0, 1] / critic_error.var()),
            "corr_by_h": {f"{h}": float(np.corrcoef(estimate, value["a0"])[0, 1])
                          for h, value in advantages_by_h.items()},
            "sd": float(np.sqrt(variance)),
        }
    return {"reliability": float(reliability), "critic_error_sd": float(critic_error.std()),
            "critic_error_mean": float(critic_error.mean()), "a_true_sd": float(truth.std()), "lambdas": table}


# ---------------------------------------------------------------- update ceiling

def expand_fresh(batch, natives, logprobs, advantages):
    """(s, k) rows for the K fresh actions: the per-state oracle surrogate's batch."""
    count = natives.shape[1]
    repeat = lambda value: value.repeat_interleave(count, 0)  # noqa: E731
    return {"obs": repeat(batch["obs"]), "alpha": repeat(batch["alpha"]), "beta": repeat(batch["beta"]),
            "native": natives.flatten(0, 1), "logprob": logprobs.flatten(), "a_true": advantages}


def oracle_gain(agent, part):
    """Oracle dJ = mean_s mean_k (r(a_k | s) - 1) A_true(s, a_k), with its per-state SE."""
    _, _, logratio = bench.logratio_of(agent, part)
    per_state = (logratio.expm1().view_as(part["a_true"]) * part["a_true"]).mean(1)
    return torch.stack((per_state.mean(), per_state.std() / per_state.numel() ** 0.5))


def update_ceiling(agent, snapshot, cli, train_batches, train_fresh, eval_bench, eval_fresh):
    args = bench.snapshot_args(snapshot)
    all_rules = bench.default_rules(args, (1.0,))
    reference = all_rules[0]
    if reference.name != "ppo" or set(reference.config) != {"clip", "clip_upper", "norm_adv"}:
        raise RuntimeError(f"unexpected reference rule {reference.label}")
    native = next(rule for rule in all_rules if rule.family == bench.native_family(snapshot, all_rules))
    views = {"ppo": {}, "ppo_lam0.8": {"lam": 0.8}, "ppo_lam0.5": {"lam": 0.5},
             "oracle1": {"view": "oracle1"}, "oracleK": {"view": "oracleK"}}
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit, 512)
    base_state = copy.deepcopy(agent.state_dict())
    evaluator = torch.compile(lambda part: bench.evaluate_part(agent, part), fullgraph=True, dynamic=False)
    gain = torch.compile(lambda part: oracle_gain(agent, part), fullgraph=True, dynamic=False)
    names = [name for name, _ in agent.actor.named_parameters()]
    real = snapshot["actor_adam"]
    device = eval_bench["obs"].device
    real_step = real[names[0]]["step"].to(device=device, dtype=torch.float32)
    real_first = [real[name]["exp_avg"].to(device) for name in names]
    real_second = [real[name]["exp_avg_sq"].to(device) for name in names]

    def compiled_loss(rule):
        return torch.compile(lambda batch, targets: rule.loss(agent, batch, targets, rule.config),
                             fullgraph=True, dynamic=False)

    native_loss = compiled_loss(native)
    native_targets = [native.prepare(batch[native.advantage_key], native.config, batch) for batch in train_batches]
    native_power = bench.gradient_power(agent, base_state, native, train_batches, native_targets, native_loss)
    results, frontier = {}, {}
    for name, extra in views.items():
        rule = bench.Rule("ppo", bench.ppo_loss, bench.prepare_ppo, {**reference.config, **extra}, 1.0,
                          reference.max_grad_norm, label=name)
        batches = train_fresh if name == "oracleK" else train_batches
        loss_fn = compiled_loss(rule)
        targets = [rule.prepare(batch[rule.advantage_key], rule.config, batch) for batch in batches]
        # Every rule steps in the trainer's per-parameter geometry: second moments rescaled to its own
        # gradient power relative to the rule that produced them, no inherited momentum (as the bench).
        scale = bench.gradient_power(agent, base_state, rule, batches, targets, loss_fn) / native_power
        moments = (real_step, [torch.zeros_like(first) for first in real_first],
                   [second * scale for second in real_second])
        sweep = []
        for lr_mult in cli.lr_mults:
            rows = []
            for batch, target, fresh in zip(batches, targets, train_fresh):
                bench.run_update(agent, base_state, rule, batch, target, snapshot["learning_rate"] * lr_mult,
                                 moments, cli.epochs, loss_fn)
                with torch.no_grad():
                    held = bench.evaluate(evaluator, eval_bench, cli.eval_chunk)
                    odj, odj_se = gain(eval_fresh).tolist()
                    odj_train = float(gain(fresh)[0])
                rows.append({key: held[key] for key in ("dj", "dj_mc", "dj_se", "kl", "kl_loc", "kl_conc", "dh")}
                            | {"odj": odj, "odj_se": odj_se, "odj_train": odj_train})
            point = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            point["odj_rows"] = [row["odj"] for row in rows]
            point["lr_mult"] = lr_mult
            sweep.append(point)
            print(f"  {name:11s} lr x{lr_mult:<4g} KL={point['kl']:.4f} (loc {point['kl_loc']:.4f} conc "
                  f"{point['kl_conc']:.4f}) dH={point['dh']:+.4f} oracle dJ={point['odj'] * 1e3:+.3f}e-3 "
                  f"(se {point['odj_se'] * 1e3:.3f}, train {point['odj_train'] * 1e3:+.3f}) "
                  f"bench dJ={point['dj'] * 1e3:+.3f}e-3 dJmc={point['dj_mc'] * 1e3:+.3f}e-3", flush=True)
        results[name] = {"v_scale": scale, "sweep": sweep}
        frontier[name] = {f"{kl:g}": {key: bench.matched([(point["kl"], point[key]) for point in sweep], kl)
                                      for key in ("odj", "dj", "dj_mc", "dh", "odj_train")}
                          for kl in cli.kl_targets}
    agent.load_state_dict(base_state)
    ratios = {}
    for kl in cli.kl_targets:
        key = f"{kl:g}"
        ppo = frontier["ppo"][key]["odj"]
        ratios[key] = {name: (None if ppo in (None, 0) or cell[key]["odj"] is None else cell[key]["odj"] / ppo)
                       for name, cell in frontier.items()}
    return {"results": results, "frontier": frontier, "odj_ratio_to_ppo": ratios}


# ---------------------------------------------------------------- saved data (scripts/awr_advantage_model_bench.py)

CHUNK_KEYS = ("obs", "native", "alpha", "beta", "logprob", "value", *(f"advantages_{lam:g}" for lam in LAMBDAS))


def chunk_payload(batch):
    """One trainer batch on the host: flat (step, env) samples plus the (T, N) GAE inputs."""
    return {**{key: batch[key].cpu() for key in CHUNK_KEYS},
            "trace": {key: value.cpu() for key, value in batch["trace"].items()}}


def save_data(path, header, oracle, args, eval_triplet, eval_adv, eval_index, eval_chunks, model_chunks):
    """EVAL oracle states (action 0 = the sampled one, 1..K fresh) with A_true and SE per checkpoint, the
    EVAL chunks they were drawn from (eval_index is into their concatenation), and the model chunks."""
    data, natives, logprobs = eval_triplet
    evaluation = {key: data[key].cpu() for key in CHUNK_KEYS}
    evaluation["row"] = torch.as_tensor(data["host"]["row"])
    evaluation["native_all"] = torch.cat([data["native"][:, None], natives], 1).cpu()
    evaluation["logprob_all"] = torch.cat([data["logprob"][:, None], logprobs], 1).cpu()
    evaluation["index"] = torch.as_tensor(eval_index)
    for h, adv in eval_adv.items():
        evaluation[f"a_true_{h}"] = torch.as_tensor(np.concatenate([adv["a0"][:, None], adv["fresh"]], 1))
        evaluation[f"a_true_se_{h}"] = torch.as_tensor(np.concatenate([adv["a0_se"][:, None], adv["fresh_se"]], 1))
        evaluation[f"v_true_{h}"] = torch.as_tensor(adv["v"])
        evaluation[f"within_var_{h}"] = adv["within_var"]
        evaluation[f"precision_{h}"] = precision(adv)
    meta = {**header, "gamma": oracle.gamma, "lambdas": LAMBDAS, "num_envs": args.num_envs, "num_steps": args.num_steps}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "eval": evaluation, "eval_chunks": [chunk_payload(batch) for batch in eval_chunks],
                "model_chunks": [chunk_payload(batch) for batch in model_chunks]}, path)


# ---------------------------------------------------------------- driver

def check_snapshot(snapshot, epochs):
    args = bench.snapshot_args(snapshot)
    if args.env_id != ENV_ID:
        raise ValueError(f"the oracle simulator is {ENV_ID}-only; snapshot is {args.env_id}")
    if args.num_minibatches != 1 or args.update_epochs != epochs or args.gae_lambda != 0.95:
        raise ValueError("the ceiling replays 1 minibatch x --epochs full-batch epochs on gae_lambda=0.95 targets")
    if getattr(args, "ent_coef", 0.0) != 0.0 or "actor_adam" not in snapshot:
        raise ValueError("snapshot needs actor Adam state and ent_coef=0")
    return args


def run_snapshot(path, cli, device, model_module, pool):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    args = check_snapshot(snapshot, cli.epochs)
    torch._dynamo.reset()  # graphs close over the previous snapshot's agent
    envs = model_module.make_training_env(args, "oracle_probe")
    agent = model_module.Agent(envs).to(device)
    envs.close()
    agent.load_state_dict(snapshot["model"])
    agent.eval()
    physics = Physics(cli.threads, cli.seed)
    oracle = Oracle(agent, snapshot, physics, cli, device)
    header = {"snapshot": str(path), "trainer": snapshot["trainer_module"], "global_step": snapshot["global_step"],
              "K": cli.fresh, "M": cli.rollouts, "H": cli.horizon, "checkpoints": oracle.checkpoints}
    print(json.dumps(header), flush=True)

    start = time.perf_counter()
    validation = {"sampler": validate_sampler(device, oracle.seed_hash, cli.candidates),
                  "physics": validate_physics(oracle, cli.seed)}
    ours = policy_episode_returns(oracle, cli.validate_episodes)
    _, reference = bench.collect(model_module, agent, snapshot, cli.reference_chunks, device, cli.seed)
    reference = np.asarray(reference)
    z = (ours.mean() - reference.mean()) / math.sqrt(ours.var(ddof=1) / len(ours) + reference.var(ddof=1) / len(reference))
    validation["returns"] = {"simulator_mean": float(ours.mean()), "simulator_sd": float(ours.std()), "simulator_n": len(ours),
                             "trainer_mean": float(reference.mean()), "trainer_sd": float(reference.std()),
                             "trainer_n": len(reference), "z": float(z)}
    print(f"validation: sampler KS ok (max stat {max(v['ks'] for k, v in validation['sampler'].items() if isinstance(v, dict)):.4f}, "
          f"coupling corr {validation['sampler']['coupling_corr_5pct']:.3f}); physics bitwise vs gymnasium over "
          f"{TIME_LIMIT} steps x {validation['physics']['copies']} copies, reward err "
          f"{validation['physics']['max_reward_error']:.1e}; returns {ours.mean():.1f}±{ours.std():.1f} (n {len(ours)}) vs "
          f"trainer {reference.mean():.1f}±{reference.std():.1f} (n {len(reference)}) z={z:+.2f} "
          f"[{time.perf_counter() - start:.0f}s]", flush=True)
    assert abs(z) < 4.0, f"frozen-policy return mismatch: z={z:.2f}"

    start = time.perf_counter()
    # Time order: oracle TRAIN chunks, EVAL chunks, then the model-fitting chunks (disjoint from both).
    batches = collect(oracle, cli.train_chunks + cli.eval_chunks + cli.model_chunks, args.num_envs, args.num_steps,
                      cli.seed)
    train_batches = batches[:cli.train_chunks]
    eval_chunks = batches[cli.train_chunks:cli.train_chunks + cli.eval_chunks]
    model_chunks = batches[cli.train_chunks + cli.eval_chunks:]
    eval_all = merge(eval_chunks)
    eval_index = np.sort(np.random.default_rng(cli.seed).choice(len(eval_all["host"]["sid"]), cli.eval_states,
                                                                replace=False))
    eval_set = select(eval_all, eval_index)
    n_train = cli.train_chunks * args.num_envs * args.num_steps
    n_eval = len(eval_set["host"]["sid"])
    rate = validation["physics"]["env_steps_per_s"]
    total = (n_train + n_eval) * (cli.fresh + 1) * cli.rollouts * cli.horizon
    print(f"collected {len(batches)} chunks in {time.perf_counter() - start:.0f}s; oracle: {n_train} train + {n_eval} eval "
          f"states, {total / 1e6:.0f}M env steps, ~{total / rate / 60:.1f} min at {rate / 1e6:.2f}M steps/s", flush=True)

    start = time.perf_counter()
    sets = {}
    if train_batches:
        train = merge(train_batches)
        sets["train"] = (train, *fresh_actions(oracle, train, cli.fresh))
    sets["eval"] = (eval_set, *fresh_actions(oracle, eval_set, cli.fresh))
    advantages = {}
    for name, (data, natives, _) in sets.items():
        host = data["host"]
        returns = oracle_returns(oracle, host["state"], host["age"], host["row"], host["sid"],
                                 torch.cat([data["native"][:, None], natives], 1), cli.group_states, pool, name)
        advantages[name] = {h: oracle_advantages(q, 1) for h, q in returns.items()}
    oracle_s = time.perf_counter() - start
    precision_table = {f"{h}": {name: precision(adv[h]) for name, adv in advantages.items()} for h in oracle.checkpoints}
    print(f"oracle precision ({oracle_s:.0f}s): set H | A(a0) sd, SE rms, noise share | fresh sd, SE rms, noise share | "
          f"within-state sd | CRN gain", flush=True)
    for h, cells in precision_table.items():
        for name, cell in cells.items():
            a0, fresh = cell["a0"], cell["fresh"]
            print(f"  {name:5s} {h:>4} | {a0['sd']:.4f} {a0['se_rms']:.4f} {a0['noise_share']:.3f} | {fresh['sd']:.4f} "
                  f"{fresh['se_rms']:.4f} {fresh['noise_share']:.3f} | {cell['within_sd']:.4f} | {cell['crn_gain']:.1f}",
                  flush=True)

    # Anatomy on the oracle TRAIN states when there are any, else on EVAL (both are sampled-action sets).
    anatomy_name = "train" if "train" in sets else "eval"
    anatomy_data = sets[anatomy_name][0]
    with torch.no_grad():
        features = torch.cat([anatomy_data["obs"], trunk_features(agent.actor, anatomy_data["obs"]),
                              trunk_features(agent.critic, anatomy_data["obs"])], 1).double().cpu().numpy()
    primary = cli.rule_horizon
    anatomy_table = anatomy(anatomy_data, advantages[anatomy_name], primary, features)
    anatomy_table["set"] = anatomy_name
    print(f"anatomy vs A_true(H={primary}) on {len(features)} {anatomy_name} sampled actions (reliability "
          f"{anatomy_table['reliability']:.3f}, critic error V^pi - V: mean {anatomy_table['critic_error_mean']:+.4f} "
          f"sd {anatomy_table['critic_error_sd']:.4f}):", flush=True)
    print("  lambda | corr  R2   R2sig | slope | within/VarHat | err share | offset share (R2 of err) | "
          "corr(err, dV) | corr by H", flush=True)
    for lam, row in anatomy_table["lambdas"].items():
        print(f"  {lam:>6} | {row['corr']:.3f} {row['r2']:.3f} {row['r2_signal']:.3f} | {row['slope_true_on_hat']:.3f} | "
              f"{row['within_true_over_var_hat']:.4f} | {row['error_var_share']:.3f} | {row['offset_share']:.3f} "
              f"({row['offset_r2_of_error']:.3f}) | {row['corr_error_critic']:+.3f} | "
              + " ".join(f"{h}:{value:.3f}" for h, value in row["corr_by_h"].items()), flush=True)

    if cli.save_data is not None:
        path_out = cli.save_data / f"{Path(path).parent.name}_{snapshot['global_step']}.pt"
        save_data(path_out, header, oracle, args, sets["eval"], advantages["eval"], eval_index, eval_chunks,
                  model_chunks)
        print(f"saved {path_out}", flush=True)

    ceiling = None
    if not cli.skip_ceiling:
        if "train" not in sets:
            raise ValueError("the update ceiling needs --train-chunks > 0")
        train, train_natives, train_logprobs = sets["train"]
        train_adv, eval_adv = advantages["train"], advantages["eval"]
        _, eval_natives, eval_logprobs = sets["eval"]
        # Update ceiling on the TRAIN batches (one per chunk), scored on EVAL.
        chunk_size = args.num_envs * args.num_steps
        a0_true = torch.as_tensor(train_adv[primary]["a0"], dtype=torch.float32, device=device)
        fresh_true = torch.as_tensor(train_adv[primary]["fresh"], dtype=torch.float32, device=device)
        fresh_parts = []
        for index, batch in enumerate(train_batches):
            part = slice(index * chunk_size, (index + 1) * chunk_size)
            std = batch["advantages_0.95"].std()
            batch["advantages_oracle1"] = a0_true[part] * (std / a0_true[part].std())
            fresh = expand_fresh(batch, train_natives[part], train_logprobs[part], fresh_true[part])
            fresh["advantages_oracleK"] = fresh["a_true"].flatten() * (std / fresh["a_true"].std())
            fresh_parts.append(fresh)
        eval_bench = {key: eval_all[key] for key in ("obs", "native", "logprob", "alpha", "beta", "advantages",
                                                     "advantages_mc")}
        eval_bench["zbin"] = torch.bucketize(bench.zscore(eval_bench["advantages"]),
                                             torch.tensor(bench.Z_EDGES, device=device))
        eval_fresh = expand_fresh(eval_set, eval_natives, eval_logprobs,
                                  torch.as_tensor(eval_adv[primary]["fresh"], dtype=torch.float32, device=device))
        print(f"update ceiling: {len(train_batches)} train batches, eval {eval_bench['obs'].shape[0]} samples (bench dJ) + "
              f"{n_eval} states x {cli.fresh} actions (oracle dJ)", flush=True)
        ceiling = update_ceiling(agent, snapshot, cli, [
            {key: value for key, value in batch.items() if key not in ("host", "trace")} for batch in train_batches],
            fresh_parts, eval_bench, eval_fresh)
        print("oracle dJ (e-3) at matched KL [bench dJ / dJmc]:", flush=True)
        for name, cells in ceiling["frontier"].items():
            print(f"  {name:11s} " + "  ".join(
                f"KL {kl}: " + ("-" if cell["odj"] is None else
                                f"{cell['odj'] * 1e3:+.3f} [{cell['dj'] * 1e3:+.3f} / {cell['dj_mc'] * 1e3:+.3f}]")
                for kl, cell in cells.items()), flush=True)
        for kl, ratio in ceiling["odj_ratio_to_ppo"].items():
            print(f"  ratio to ppo at KL {kl}: " + " ".join(
                f"{name} {'-' if value is None else f'{value:.2f}'}" for name, value in ratio.items()), flush=True)
    return {"header": header, "validation": validation, "precision": precision_table, "anatomy": anatomy_table,
            "ceiling": ceiling, "oracle_seconds": oracle_s}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--fresh", type=int, default=7, help="K fresh actions per state")
    parser.add_argument("--rollouts", type=int, default=4, help="M CRN rollouts per (state, action)")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--rule-horizon", type=int, default=200, help="the A_true horizon for anatomy and rules")
    parser.add_argument("--candidates", type=int, default=5, help="Marsaglia-Tsang candidates per Gamma draw")
    parser.add_argument("--train-chunks", type=int, default=2)
    parser.add_argument("--eval-chunks", type=int, default=16)
    parser.add_argument("--eval-states", type=int, default=16384)
    parser.add_argument("--model-chunks", type=int, default=0, help="chunks after EVAL saved for model fitting")
    parser.add_argument("--save-data", type=Path, default=None, help="directory for per-snapshot data files")
    parser.add_argument("--skip-ceiling", action="store_true")
    parser.add_argument("--group-states", type=int, default=2048, help="states per branch group (x (K+1) M branches)")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--validate-episodes", type=int, default=256)
    parser.add_argument("--reference-chunks", type=int, default=24, help="trainer-pipeline chunks for the return check")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-chunk", type=int, default=65536)
    parser.add_argument("--lr-mults", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.5, 6.5, 10.0])
    parser.add_argument("--kl-targets", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    cli = parser.parse_args()
    if not all(1 <= h <= cli.horizon for h in cli.checkpoints) or cli.rule_horizon not in set(cli.checkpoints) | {cli.horizon}:
        raise ValueError("--checkpoints must lie in [1, --horizon] and --rule-horizon must be one of them")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_module = importlib.import_module(bench.MODEL_MODULE)
    model_module.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(cli.seed)
    device = torch.device("cuda")
    report = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for path in cli.snapshots:
            report.append(run_snapshot(path, cli, device, model_module, pool))
            if cli.out is not None:
                cli.out.parent.mkdir(parents=True, exist_ok=True)
                cli.out.write_text(json.dumps(report, indent=1, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
