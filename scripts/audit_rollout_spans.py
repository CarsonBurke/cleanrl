"""Reconcile v9's production PhaseTimer spans against per-region host cost.

``scripts/benchmark_rollout_chain.py`` measures each link of the rollout chain
in a *tight loop over one link*. The live trainer
(``cleanrl/plasticity/ppo_continuous_action_sphere_sdplast_v9.py``) reports
144us/step of ``rollout`` and 37us/step of ``normalize_transfer`` against
70.7us + 13.6us and 3.7us + 1.3us of tight-loop components. This script
replays v9's rollout loop body verbatim (same order, same spans, same objects,
same shapes) and attributes the wall time region by region, so both spans are
explained rather than estimated.

Three deliberate controls separate *work* from *context*:

* ``--mode spans-off`` removes the three ``PhaseTimer`` spans, isolating
  instrumentation cost at the real call frequency.
* ``--mode fused`` swaps ``HostSiTUSphereActor`` for ``make_host_mirror``
  (native fused graph) with everything else identical.
* every step re-runs the normalize/transfer calls a second time against
  duplicate state ("warm" rows). The duplicate does identical work on
  identical shapes, but with caches already primed by the first pass, so the
  first-pass/second-pass delta measures how much of a span is cold-cache and
  post-wake overhead rather than arithmetic.

Nothing here mutates the trainer or ``cleanrl/shared``; it only imports them.
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

_BLOAT = None  # keeps the --bloat object graph alive for the whole run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# CLEANRL_ENV_SPIN is read when the physics pool is created, so it must be set
# before make_mujoco_vector_env. Parse it out of argv early.
_SPIN = None
for _i, _a in enumerate(sys.argv):
    if _a == "--env-spin" and _i + 1 < len(sys.argv):
        _SPIN = sys.argv[_i + 1]
    elif _a.startswith("--env-spin="):
        _SPIN = _a.split("=", 1)[1]
if _SPIN is not None:
    os.environ["CLEANRL_ENV_SPIN"] = _SPIN

import torch  # noqa: E402
from torch import nn  # noqa: E402

from cleanrl.shared.host_actor import HostSiTUSphereActor, make_situ_sphere_trunk  # noqa: E402
from cleanrl.shared.host_graph import make_host_mirror  # noqa: E402
from cleanrl.shared.mujoco_env import make_mujoco_vector_env  # noqa: E402
from cleanrl.shared.ppo_loop import TruncationBootstrapCache  # noqa: E402
from cleanrl.shared.rollout_transfer import RolloutTransfer  # noqa: E402
from cleanrl.shared.runtime import configure_runtime  # noqa: E402
from cleanrl.shared.sampling import sample_beta_actions_host  # noqa: E402
from cleanrl.shared.timing import PhaseTimer  # noqa: E402
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm  # noqa: E402

# The fine-mode probe layout. Names label the interval that ENDS at the probe.
PROBES = (
    "t_span_start",        # timer.start("rollout")
    "t_policy_forward",    # host_actor(obs_step)
    "t_beta_sample",       # sample_beta_actions_host(...)
    "t_isfinite",          # np.isfinite(action).all()
    "t_reshape",           # action.reshape(...)
    "t_span_stop",         # timer.stop()   [end of rollout span]
    "t_env_start",         # timer.start("env")
    "t_env_step",          # envs.step(host_action)
    "t_env_stop",          # timer.stop()
    "t_nt_start",          # timer.start("normalize_transfer")
    "t_rew_norm",          # rew_norm.normalize(...)
    "t_obs_norm",          # obs_norm.normalize_step(...)
    "t_bootstraps",        # bootstraps.push_normalized(...)
    "t_transfer_push",     # transfer.push(...)
    "t_nt_stop",           # timer.stop()   [end of normalize_transfer span]
    "t_bookkeeping",       # global_step += ...; final_info loop
    "t_warm_rew",          # duplicate rew_norm.normalize
    "t_warm_obs",          # duplicate obs_norm.normalize_step
    "t_warm_boot",         # duplicate bootstraps.push_normalized
    "t_warm_push",         # duplicate transfer.push
    "t_warm_policy",       # duplicate host_actor(obs_step)
)

ROLLOUT_ROWS = ("t_span_start", "t_policy_forward", "t_beta_sample",
                "t_isfinite", "t_reshape", "t_span_stop")
NT_ROWS = ("t_nt_start", "t_rew_norm", "t_obs_norm", "t_bootstraps",
           "t_transfer_push", "t_nt_stop")


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def probe_overhead(samples=20001):
    """Median cost of one ``perf_counter`` call, measured the way we use it."""
    deltas = np.empty(samples - 1)
    stamps = np.empty(samples)
    for i in range(samples):
        stamps[i] = time.perf_counter()
    np.subtract(stamps[1:], stamps[:-1], out=deltas)
    return float(np.median(deltas))


def build(args):
    """Everything v9's main() builds before its rollout loop, same order."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    device = torch.device("cuda")
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend="native",
                                  num_threads=min(args.env_threads, args.num_envs))
    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_shape = tuple(envs.single_action_space.shape)
    act_dim = int(np.prod(action_shape))

    actor = None
    if not args.synthetic_actor:
        try:  # the real trainer's module tree, plastic wrappers included
            from cleanrl.plasticity.ppo_continuous_action_sphere_sdplast_v9 import Agent, Args
            trainer_args = Args(num_envs=args.num_envs, num_steps=args.num_steps)
            actor = Agent(envs, trainer_args).to(device).actor
            source = "v9 Agent.actor (PlasticLinear sites live)"
        except Exception as error:  # pragma: no cover - diagnostic path
            print(f"  [warn] could not build the v9 Agent ({error!r}); using a bare trunk")
    if actor is None:
        trunk = make_situ_sphere_trunk(obs_dim, args.width, args.n_blocks).to(device)
        head = nn.Linear(args.width, 2 * act_dim).to(device)
        actor = nn.Sequential(trunk, head)
        source = "make_situ_sphere_trunk + Linear head"

    if args.mode == "fused":
        host_actor = make_host_mirror(actor, args.num_envs)
    else:
        host_actor = HostSiTUSphereActor(actor, args.num_envs)

    low = np.asarray(envs.single_action_space.low, dtype=np.float32).reshape(-1).copy()
    high = np.asarray(envs.single_action_space.high, dtype=np.float32).reshape(-1).copy()
    sampler = np.random.default_rng(args.seed)

    fields = {"observations": obs_shape, "native_actions": (act_dim,)}
    ctx = dict(
        envs=envs, device=device, obs_shape=obs_shape, action_shape=action_shape,
        act_dim=act_dim, host_actor=host_actor, mirror=type(host_actor).__name__,
        actor_source=source, low=low, high=high, sampler=sampler,
        transfer=RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                 non_blocking=False, fields=fields),
        bootstraps=TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape),
        obs_norm=VectorObsNorm(args.num_envs, obs_shape),
        rew_norm=VectorRewardNorm(args.num_envs, args.gamma),
        # Duplicate state for the warm second pass: same shapes, same code, but
        # entered with the caches already hot from the first pass.
        transfer2=RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                  non_blocking=False, fields=fields),
        bootstraps2=TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape),
        obs_norm2=VectorObsNorm(args.num_envs, obs_shape),
        rew_norm2=VectorRewardNorm(args.num_envs, args.gamma),
        timer=PhaseTimer(),
        suppress=np.zeros(args.num_envs, dtype=bool),
    )
    return ctx


def run_fine(args, ctx):
    """v9's loop body with a probe between every statement."""
    envs, timer = ctx["envs"], ctx["timer"]
    host_actor, sampler = ctx["host_actor"], ctx["sampler"]
    low, high = ctx["low"], ctx["high"]
    obs_norm, rew_norm = ctx["obs_norm"], ctx["rew_norm"]
    bootstraps, transfer = ctx["bootstraps"], ctx["transfer"]
    obs_norm2, rew_norm2 = ctx["obs_norm2"], ctx["rew_norm2"]
    bootstraps2, transfer2 = ctx["bootstraps2"], ctx["transfer2"]
    suppress = ctx["suppress"]
    num_envs, num_steps = args.num_envs, args.num_steps
    reshape_to = (num_envs,) + ctx["action_shape"]
    spans = args.mode != "spans-off"

    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs_np = obs_norm.normalize(raw_obs)
    obs_norm2.normalize(raw_obs)
    global_step = 0
    perf = time.perf_counter
    samples = []
    boundary = []
    total = args.warmup + args.steps

    cpu0 = cpu_seconds()
    wall0 = perf()
    for i in range(total):
        step = i % num_steps
        t0 = perf()
        if spans:
            timer.start("rollout", use_cuda=False)
        t1 = perf()
        obs_step = next_obs_np
        logits = host_actor(obs_step)
        t2 = perf()
        native, action = sample_beta_actions_host(logits, low, high, sampler)
        t3 = perf()
        ok = np.isfinite(action).all()
        t4 = perf()
        host_action = action.reshape(reshape_to)
        t5 = perf()
        if spans:
            timer.stop()
        t6 = perf()
        if spans:
            timer.start("env", use_cuda=False)
        t7 = perf()
        raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
        t8 = perf()
        if spans:
            timer.stop()
        t9 = perf()
        if spans:
            timer.start("normalize_transfer", use_cuda=False)
        t10 = perf()
        reward = rew_norm.normalize(raw_reward, terms)
        t11 = perf()
        next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
        t12 = perf()
        bootstraps.push_normalized(step, truncs, transition_obs)
        t13 = perf()
        transfer.push(step, reward, terms, truncs,
                      observations=obs_step, native_actions=native)
        t14 = perf()
        if spans:
            timer.stop()
        t15 = perf()
        global_step += num_envs
        episodes = 0
        for index, info in enumerate(infos.get("final_info", ())):
            if info and "episode" in info:
                if suppress[index]:
                    suppress[index] = False
                    continue
                float(info["episode"]["r"])
                float(info["episode"]["l"])
                episodes += 1
        t16 = perf()
        # --- warm second pass: identical work, primed caches ---
        reward2 = rew_norm2.normalize(raw_reward, terms)
        t17 = perf()
        _, transition2 = obs_norm2.normalize_step(raw_obs, terms, truncs, infos)
        t18 = perf()
        bootstraps2.push_normalized(step, truncs, transition2)
        t19 = perf()
        transfer2.push(step, reward2, terms, truncs,
                       observations=obs_step, native_actions=native)
        t20 = perf()
        host_actor(obs_step)
        t21 = perf()
        if not ok:
            raise FloatingPointError("policy produced nonfinite actions")
        if i >= args.warmup:
            samples.append((t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10,
                            t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21))
            boundary.append(bool(np.any(terms) or np.any(truncs)) or episodes > 0)
    wall = perf() - wall0
    cpu = cpu_seconds() - cpu0

    extras = {}
    if args.extras:
        # Isolate the rollout rows that dominate, using the loop's own live
        # arrays, so the numbers are directly comparable to the rows above.
        logits = host_actor(next_obs_np)
        half = logits.shape[-1] // 2
        conc = np.logaddexp(0.0, logits, dtype=np.float32) + 1.0
        alpha, beta = conc[..., :half], conc[..., half:]

        def bench(fn, iters=4000):
            for _ in range(200):
                fn()
            samples = []
            for _ in range(iters):
                start = perf()
                fn()
                samples.append(perf() - start)
            return float(np.median(samples)) * 1e6

        extras["mirror forward"] = bench(lambda: host_actor(next_obs_np))
        extras["rng.beta only"] = bench(lambda: sampler.beta(alpha, beta))
        extras["beta_sample total"] = bench(
            lambda: sample_beta_actions_host(logits, low, high, sampler))
        extras["logaddexp only"] = bench(
            lambda: np.logaddexp(0.0, logits, dtype=np.float32))
        extras["isfinite+all"] = bench(lambda: np.isfinite(logits).all())
        extras["one np.add (dispatch)"] = bench(
            lambda: np.add(conc, 1.0, out=conc))
        extras["PhaseTimer start+stop"] = bench(
            lambda: (timer.start("probe", use_cuda=False), timer.stop()))

    stamps = np.asarray(samples)
    return {
        "stamps": stamps,
        "boundary": np.asarray(boundary),
        "wall_per_step_us": wall / total * 1e6,
        "cpu_per_step_us": cpu / total * 1e6,
        "timer": timer.summary() if spans else {},
        "global_step": global_step,
        "extras": extras,
    }


def run_coarse(args, ctx):
    """v9's loop body byte-for-byte (spans only), for the ground-truth spans."""
    envs, timer = ctx["envs"], ctx["timer"]
    host_actor, sampler = ctx["host_actor"], ctx["sampler"]
    low, high = ctx["low"], ctx["high"]
    obs_norm, rew_norm = ctx["obs_norm"], ctx["rew_norm"]
    bootstraps, transfer = ctx["bootstraps"], ctx["transfer"]
    suppress = ctx["suppress"]
    num_envs, num_steps = args.num_envs, args.num_steps
    action_shape = ctx["action_shape"]
    spans = args.mode != "spans-off"

    def act(observations):
        native, action = sample_beta_actions_host(
            host_actor(observations), low, high, sampler)
        if not np.isfinite(action).all():
            raise FloatingPointError("policy produced nonfinite actions")
        return native, action.reshape((num_envs,) + action_shape)

    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs_np = obs_norm.normalize(raw_obs)
    global_step = 0
    perf = time.perf_counter
    starts = []
    total = args.warmup + args.steps
    cpu0 = cpu_seconds()
    wall0 = perf()
    for i in range(total):
        step = i % num_steps
        starts.append(perf())
        if spans:
            with timer.span("rollout", use_cuda=False):
                obs_step = next_obs_np
                native, host_action = act(obs_step)
            with timer.span("env", use_cuda=False):
                raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
            with timer.span("normalize_transfer", use_cuda=False):
                reward = rew_norm.normalize(raw_reward, terms)
                next_obs_np, transition_obs = obs_norm.normalize_step(
                    raw_obs, terms, truncs, infos)
                bootstraps.push_normalized(step, truncs, transition_obs)
                transfer.push(step, reward, terms, truncs,
                              observations=obs_step, native_actions=native)
        else:
            obs_step = next_obs_np
            native, host_action = act(obs_step)
            raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
            reward = rew_norm.normalize(raw_reward, terms)
            next_obs_np, transition_obs = obs_norm.normalize_step(
                raw_obs, terms, truncs, infos)
            bootstraps.push_normalized(step, truncs, transition_obs)
            transfer.push(step, reward, terms, truncs,
                          observations=obs_step, native_actions=native)
        global_step += num_envs
        for index, info in enumerate(infos.get("final_info", ())):
            if info and "episode" in info:
                if suppress[index]:
                    suppress[index] = False
                    continue
                float(info["episode"]["r"])
                float(info["episode"]["l"])
    wall = perf() - wall0
    cpu = cpu_seconds() - cpu0
    starts = np.asarray(starts[args.warmup:])
    steps = np.diff(starts) * 1e6
    return {
        "wall_per_step_us": wall / total * 1e6,
        "cpu_per_step_us": cpu / total * 1e6,
        "step_median_us": float(np.median(steps)),
        "step_min_us": float(np.min(steps)),
        "timer": timer.summary() if spans else {},
    }


def report_fine(args, ctx, result, overhead_us):
    stamps = result["stamps"]
    deltas = np.diff(stamps, axis=1) * 1e6  # (steps, len(PROBES))
    index = {name: i for i, name in enumerate(PROBES)}
    median = {name: float(np.median(deltas[:, i])) for name, i in index.items()}
    mean = {name: float(np.mean(deltas[:, i])) for name, i in index.items()}
    p90 = {name: float(np.percentile(deltas[:, i], 90)) for name, i in index.items()}
    p99 = {name: float(np.percentile(deltas[:, i], 99)) for name, i in index.items()}
    boundary = result["boundary"]
    clean = ~boundary
    median_clean = {name: float(np.median(deltas[clean, i])) for name, i in index.items()}

    print(f"\n  probes: {len(stamps)} steps, "
          f"{int(boundary.sum())} with a termination/truncation")
    print(f"  perf_counter call cost: {overhead_us * 1000:.1f}ns "
          f"(subtracted from every row below)")
    print(f"  {'region':<20s} {'median':>9s} {'mean':>9s} {'p90':>9s} "
          f"{'p99':>9s} {'no-bnd':>9s}")
    for name in PROBES:
        adj = max(median[name] - overhead_us, 0.0)
        print(f"  {name:<20s} {adj:9.2f} {mean[name] - overhead_us:9.2f} "
              f"{p90[name] - overhead_us:9.2f} {p99[name] - overhead_us:9.2f} "
              f"{median_clean[name] - overhead_us:9.2f}")

    def total(names, table):
        return sum(max(table[n] - overhead_us, 0.0) for n in names)

    def span_series(names):
        columns = [index[n] for n in names]
        return deltas[:, columns].sum(axis=1) - overhead_us * len(names)

    rollout = total(ROLLOUT_ROWS, median)
    nt = total(NT_ROWS, median)
    warm_rows = ("t_warm_rew", "t_warm_obs", "t_warm_boot", "t_warm_push")
    warm_nt = total(warm_rows, median)
    warm_policy = max(median["t_warm_policy"] - overhead_us, 0.0)
    env = max(median["t_env_step"] - overhead_us, 0.0)
    book = max(median["t_bookkeeping"] - overhead_us, 0.0)

    print(f"\n  {'span':<34s} {'median':>9s} {'MEAN':>9s}   "
          f"[what the trainer logs is the MEAN]")
    for label, names, production in (
        ("rollout", ROLLOUT_ROWS, 143.8),
        ("normalize_transfer", NT_ROWS, 36.8),
        ("warm rerun of normalize_transfer", warm_rows, None),
    ):
        series = span_series(names)
        note = "" if production is None else f"   production {production:.1f}"
        print(f"  {label:<34s} {np.median(series):9.2f} {series.mean():9.2f}{note}")
    print(f"  {'env step':<34s} {env:9.2f} "
          f"{mean['t_env_step'] - overhead_us:9.2f}   production 147.8")
    print(f"  {'post-span bookkeeping':<34s} {book:9.2f} "
          f"{mean['t_bookkeeping'] - overhead_us:9.2f}")

    # How much of each span's MEAN is contributed by its worst tail. The
    # trainer logs a sum over 2048 calls, so tail steps are fully in the number.
    print("\n  tail attribution (share of the span MEAN above its own median):")
    for label, names in (("rollout", ROLLOUT_ROWS),
                         ("normalize_transfer", NT_ROWS),
                         ("env", ("t_env_step",))):
        series = np.sort(span_series(names))
        mean_all = series.mean()
        med = np.median(series)
        cut99 = int(len(series) * 0.99)
        cut95 = int(len(series) * 0.95)
        excess_top1 = series[cut99:].sum() / len(series) - med * (len(series) - cut99) / len(series)
        excess_top5 = series[cut95:].sum() / len(series) - med * (len(series) - cut95) / len(series)
        print(f"    {label:<20s} mean {mean_all:7.2f}  median {med:7.2f}  "
              f"excess from worst 1% {excess_top1:6.2f}us  worst 5% {excess_top5:6.2f}us")

    print(f"\n  cold-entry penalty, normalize+transfer  {nt - warm_nt:+7.2f}us "
          f"(first pass {nt:.2f} vs warm rerun {warm_nt:.2f})")
    print(f"  cold-entry penalty, policy forward      "
          f"{median['t_policy_forward'] - overhead_us - warm_policy:+7.2f}us "
          f"(first pass {median['t_policy_forward'] - overhead_us:.2f} vs "
          f"rerun {warm_policy:.2f})")
    if result["timer"]:
        print("\n  PhaseTimer's own accounting (what the trainer would log):")
        for name, entry in result["timer"].items():
            per_step = entry["total_s"] / entry["calls"] * 1e6
            print(f"    {name:<20s} {per_step:8.2f}us/step  ({entry['calls']} calls)")
    if boundary.any():
        bnd = {name: float(np.median(deltas[boundary, i])) for name, i in index.items()}
        extra_nt = total(NT_ROWS, bnd) - total(NT_ROWS, median_clean)
        extra_roll = total(ROLLOUT_ROWS, bnd) - total(ROLLOUT_ROWS, median_clean)
        share = boundary.mean()
        print(f"\n  boundary steps: {share * 100:.2f}% of steps; "
              f"+{extra_nt:.1f}us normalize_transfer, +{extra_roll:.1f}us rollout each "
              f"-> +{extra_nt * share:.2f}/+{extra_roll * share:.2f}us/step amortized")
    print(f"\n  whole-loop wall {result['wall_per_step_us']:.1f}us/step, "
          f"cpu {result['cpu_per_step_us']:.1f}us/step")
    if result.get("extras"):
        print("\n  isolated components (tight loop, median):")
        for name, value in result["extras"].items():
            print(f"    {name:<26s} {value:8.2f}us")
    return {"rollout": rollout, "rollout_mean": float(span_series(ROLLOUT_ROWS).mean()),
            "env": env, "normalize_transfer": nt,
            "nt_mean": float(span_series(NT_ROWS).mean()),
            "bookkeeping": book, "warm_nt": warm_nt, "warm_policy": warm_policy,
            "policy_forward": max(median["t_policy_forward"] - overhead_us, 0.0),
            "beta_sample": max(median["t_beta_sample"] - overhead_us, 0.0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mode", default="numpy",
                        choices=("numpy", "fused", "spans-off"),
                        help="numpy = v9 as shipped; fused = make_host_mirror; "
                             "spans-off = v9 without PhaseTimer spans")
    parser.add_argument("--granularity", default="fine", choices=("fine", "coarse"))
    parser.add_argument("--extras", action="store_true",
                        help="after the loop, isolate the dominant rollout "
                             "components in a tight loop for comparison")
    parser.add_argument("--synthetic-actor", action="store_true",
                        help="skip the v9 Agent and mirror a bare sphere trunk")
    parser.add_argument("--env-spin", default=None,
                        help="CLEANRL_ENV_SPIN pause budget for the physics pool")
    parser.add_argument("--gc", default="on", choices=("on", "off", "frozen"),
                        help="on = default; off = gc.disable(); frozen = "
                             "gc.freeze() after setup (moves the static graph "
                             "out of every future collection)")
    parser.add_argument("--bloat", type=int, default=0,
                        help="GC-tracked objects to allocate before the loop, "
                             "emulating the live trainer's object graph")
    args = parser.parse_args()

    with open("/proc/loadavg") as handle:
        load = handle.read().split()[:3]
    overhead_us = probe_overhead() * 1e6
    if args.bloat:
        global _BLOAT
        _BLOAT = [{"i": i, "p": (i, i + 1)} for i in range(args.bloat)]
    ctx = build(args)
    if args.gc == "off":
        gc.disable()
    elif args.gc == "frozen":
        gc.collect()
        gc.freeze()
    tracked = len(gc.get_objects())
    gc_before = [dict(s) for s in gc.get_stats()]
    print(f"env={args.env_id} num_envs={args.num_envs} num_steps={args.num_steps} "
          f"env_threads={args.env_threads} mode={args.mode} "
          f"granularity={args.granularity}")
    print(f"  mirror={ctx['mirror']}  actor={ctx['actor_source']}")
    print(f"  loadavg={'/'.join(load)}  "
          f"CLEANRL_ENV_SPIN={os.environ.get('CLEANRL_ENV_SPIN', '<unset>')}")
    print(f"  gc={args.gc} tracked_objects={tracked} "
          f"threads={threading.active_count()}")

    summaries = []
    for repeat in range(args.repeats):
        ctx["timer"].reset()
        if args.granularity == "fine":
            result = run_fine(args, ctx)
            print(f"\n--- repeat {repeat + 1}/{args.repeats} ---")
            summaries.append(report_fine(args, ctx, result, overhead_us))
        else:
            result = run_coarse(args, ctx)
            print(f"\n--- repeat {repeat + 1}/{args.repeats} ---")
            print(f"  step median {result['step_median_us']:.1f}us  "
                  f"min {result['step_min_us']:.1f}us  "
                  f"wall {result['wall_per_step_us']:.1f}us  "
                  f"cpu {result['cpu_per_step_us']:.1f}us")
            row = {}
            for name, entry in result["timer"].items():
                per_step = entry["total_s"] / entry["calls"] * 1e6
                row[name] = per_step
                print(f"    {name:<20s} {per_step:8.2f}us/step")
            row["step_total"] = result["step_median_us"]
            summaries.append(row)

    if args.repeats > 1 and summaries:
        print("\n  medians across repeats:")
        for key in summaries[0]:
            values = [s[key] for s in summaries if key in s]
            print(f"    {key:<24s} {statistics.median(values):8.2f}us")
    gc_after = gc.get_stats()
    print("  gc collections during the whole process: " + ", ".join(
        f"gen{i} {after['collections'] - before['collections']}"
        for i, (before, after) in enumerate(zip(gc_before, gc_after))))

    ctx["transfer"].close()
    ctx["transfer2"].close()
    ctx["envs"].close()


if __name__ == "__main__":
    main()
