"""Queued fixed-work PPO optimizer comparison; this is not a learning run.

Compare compiled loss with six eager gathers against indexing inside the loss
compiler boundary. Both variants use the same FP32 model, B=32768 rollout data,
index order, clipping and fused Adam. Numerical checks span changing Adam states,
compiled inference peers, and next-rollout reuse of the same input allocations.
Submit through mlq with --max-parallel-runs 1; seed 1 is the default.
"""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.ppo_update import make_minibatch_loss
from cleanrl.shared.runtime import configure_runtime
from scripts.benchmark_ppo_base import (
    METRICS, TOLERANCES, compile_fn, difference, flat_gradients, flat_parameters,
    require_checks,
)
from scripts.benchmark_mujoco_throughput import json_compatible, timing_summary, write_scalars


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--minibatch-sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--exp-name", default="ppo_indexed_loss_execution")
    args = parser.parse_args()
    if min(args.batch_size, *args.minibatch_sizes, args.iterations, args.repeats) <= 0:
        parser.error("sizes, iterations and repeats must be positive")
    if max(args.minibatch_sizes) > args.batch_size or args.minimum_speedup <= 1:
        parser.error("minibatches must fit the rollout and minimum speedup must exceed one")
    return args


@torch.no_grad()
def make_fixtures(model, batch_size):
    fixtures = []
    for _ in range(3):
        observations = torch.randn(batch_size, 17, device="cuda")
        native = torch.rand(batch_size, 6, device="cuda") * 0.8 + 0.1
        alpha, beta, values = model.get_policy_and_value(observations)
        logprobs = model.action_logprob(alpha, beta, native)
        offsets = torch.linspace(-0.4, 0.4, batch_size, device="cuda")
        inputs = (observations, native, logprobs + offsets,
                  torch.randn(batch_size, device="cuda") * 2,
                  values.flatten() + 8 + torch.randn(batch_size, device="cuda"),
                  values.flatten() + torch.linspace(-0.5, 0.5, batch_size, device="cuda"))
        fixtures.append(tuple(x.detach().clone() for x in inputs))
    return fixtures


def make_execution(base, initial, loss_args, fixture, *, indexed):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=loss_args.learning_rate, eps=1e-5, fused=True)
    buffers = tuple(torch.empty_like(tensor) for tensor in fixture)
    peer_storage = (torch.empty_like(fixture[2]), torch.empty_like(fixture[2]))

    def raw_loss(*batch):
        return base.ppo_loss(model, *batch, loss_args)

    if indexed:
        loss_fn = make_minibatch_loss(raw_loss)
    else:
        baseline = torch.compile(raw_loss, mode="reduce-overhead", fullgraph=True, dynamic=False)

        def loss_fn(indices, *batch):
            torch.compiler.cudagraph_mark_step_begin()
            return baseline(*(tensor[indices] for tensor in batch))

    def statistics_model(observations, native):
        alpha, beta, values = model.get_policy_and_value(observations)
        return values.flatten(), model.action_logprob(alpha, beta, native)

    peer = compile_fn(statistics_model, cuda_graphs=True)

    @torch.no_grad()
    def upload_and_infer(inputs):
        # Preserve input addresses across successive rollouts. Production's
        # rollout-statistics compile disables graphs; a graph-enabled peer here
        # is a stronger lifetime check, not a requirement to clone base inputs.
        for buffer, tensor in zip(buffers, inputs):
            buffer.copy_(tensor)
        values, logprobs = peer(buffers[0], buffers[1])
        peer_storage[0].copy_(values)
        peer_storage[1].copy_(logprobs)
        # Fixed reference fixture remains identical between variants. Peer
        # outputs are observed separately and never retained as graph aliases.
        return peer_storage

    metric_storage = torch.empty(6, device="cuda")

    def step(indices, *, snapshot=False):
        loss, metrics = loss_fn(indices, *buffers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = flat_gradients(model) if snapshot else None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), loss_args.max_grad_norm)
        clipped = flat_gradients(model) if snapshot else None
        optimizer.step()
        metric_storage.copy_(metrics)
        if not snapshot:
            return None
        return {"loss": loss.detach().clone(), "metrics": metric_storage.clone(),
                "gradients": gradients, "clipped_gradients": clipped,
                "preclip_norm": norm.detach().clone(), "parameters": flat_parameters(model),
                "optimizer_state": {key: torch.cat([
                    optimizer.state[p][key].detach().reshape(-1) for p in model.parameters()
                ]).clone() for key in ("step", "exp_avg", "exp_avg_sq")}}

    return SimpleNamespace(model=model, optimizer=optimizer, upload_and_infer=upload_and_infer,
                           step=step, metrics=metric_storage)


def compare_snapshot(expected, actual):
    result = {"checks": {"finite_loss": bool(actual["loss"].isfinite()),
                         "finite_parameters": bool(actual["parameters"].isfinite().all())}}
    for key, contract in (("loss", "loss_metrics"), ("gradients", "gradients"),
                          ("clipped_gradients", "gradients"), ("preclip_norm", "gradients"),
                          ("parameters", "parameters")):
        result[key] = difference(expected[key], actual[key], contract)
    result["metrics"] = {name: difference(a, b, "loss_metrics")
                         for name, a, b in zip(METRICS, expected["metrics"], actual["metrics"])}
    result["optimizer_state"] = {key: difference(expected["optimizer_state"][key], value, "optimizer_state")
                                 for key, value in actual["optimizer_state"].items()}
    return result


def measure_pair(executions, fixtures, indices, args):
    """Alternate paired repetition order, synchronizing only at boundaries."""
    for execution in executions.values():
        execution.upload_and_infer(fixtures[0])
        for index in range(10):
            execution.step(indices[index % len(indices)])
    torch.cuda.synchronize()
    wall = {name: [] for name in executions}
    stream = {name: [] for name in executions}
    for repetition in range(args.repeats):
        names = list(executions)
        if repetition % 2:
            names.reverse()
        for name in names:
            execution = executions[name]
            execution.upload_and_infer(fixtures[repetition % len(fixtures)])
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            start.record()
            for index in range(args.iterations):
                execution.step(indices[index % len(indices)])
            end.record()
            end.synchronize()
            wall[name].append(time.perf_counter() - started)
            stream[name].append(start.elapsed_time(end) / 1000)
    result = {}
    for name in executions:
        result[name] = timing_summary(wall[name], args.iterations)
        result[name]["cuda_stream_seconds"] = stream[name]
        result[name]["cuda_stream_microseconds_per_call"] = statistics.median(stream[name]) * 1e6 / args.iterations
    return result


def benchmark_size(base, initial, fixtures, args, size, result, save):
    loss_args = base.Args()
    indices = torch.stack([torch.randperm(args.batch_size, device="cuda")[:size] for _ in range(4)])
    result.update(batch_size=args.batch_size, minibatch_size=size, rollout_rounds=3, updates_per_round=4,
                  full_input_bytes=sum(x.numel() * x.element_size() for x in fixtures[0]),
                  checks={}, updates={}, peer_inference={}, startup_seconds={})
    executions = {}
    references = []
    peers = []
    retained_metrics = {}
    for label, indexed in (("eager_indexing", False), ("compiled_indexing", True)):
        execution = executions[label] = make_execution(base, initial, loss_args, fixtures[0], indexed=indexed)
        retained_metrics[label] = torch.empty(12, 6, device="cuda")
        snapshots = result["updates"][label] = {}
        sequence = 0
        for rollout, fixture in enumerate(fixtures):
            torch.cuda.synchronize()
            started = time.perf_counter()
            peer_output = tuple(t.clone() for t in execution.upload_and_infer(fixture))
            if indexed:
                result["peer_inference"][str(rollout)] = {
                    name: difference(a, b, "inference")
                    for name, a, b in zip(("values", "logprobs"), peers[rollout], peer_output)}
            else:
                peers.append(peer_output)
            rng = torch.cuda.get_rng_state()
            for selected in indices:
                snapshot = execution.step(selected, snapshot=True)
                retained_metrics[label][sequence].copy_(snapshot["metrics"])
                if indexed:
                    snapshots[str(sequence)] = compare_snapshot(references[sequence], snapshot)
                else:
                    references.append(snapshot)
                if rollout == 0 and sequence == 0:
                    torch.cuda.synchronize()
                    result["startup_seconds"][label] = time.perf_counter() - started
                sequence += 1
            result["checks"][f"{label}_round{rollout}_rng_unchanged"] = torch.equal(torch.cuda.get_rng_state(), rng)
            require_checks(result, save)
    result["retained_metrics"] = difference(retained_metrics["eager_indexing"],
                                             retained_metrics["compiled_indexing"], "loss_metrics")
    require_checks(result, save)
    # Both variants enter timing after the same twelve ordered updates. Timing
    # performs complete optimizer steps; no isolated forward-only speed claim.
    result["full_optimizer_step"] = measure_pair(executions, fixtures, indices, args)
    for label, execution in executions.items():
        result["checks"][f"{label}_finite_after_timing"] = bool(flat_parameters(execution.model).isfinite().all())
    require_checks(result, save)
    baseline = result["full_optimizer_step"]["eager_indexing"]
    candidate = result["full_optimizer_step"]["compiled_indexing"]
    result["speedup"] = baseline["median_seconds"] / candidate["median_seconds"]
    result["paired_speedups"] = [a / b for a, b in zip(baseline["seconds"], candidate["seconds"])]
    result["adoption"] = {
        "accepted": result["speedup"] >= args.minimum_speedup and min(result["paired_speedups"]) > 1,
        "minimum_median_speedup": args.minimum_speedup,
        "requires_every_paired_repeat_faster": True,
    }
    save()


def main():
    args = parse_args()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("queued CUDA execution required; no CPU fallback")
    torch.manual_seed(args.seed)
    base = importlib.import_module("cleanrl.ppo_continuous_action")
    run_dir = Path("runs") / f"HalfCheetah-v4__{args.exp_name}__{args.seed}__{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"kind": "fixed_work_indexed_ppo_update_not_training", "status": "running", "args": vars(args),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "precision": "FP32/highest/TF32 disabled", "compile_mode": "reduce-overhead",
              "adam_fused": True, "numerical_tolerances": TOLERANCES, "minibatches": {},
              "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for path in (base.__file__, "cleanrl/shared/ppo_update.py", __file__)}}
    writer = SummaryWriter(str(run_dir))

    def save():
        (run_dir / "benchmark.json").write_text(json.dumps(json_compatible(report), indent=2, allow_nan=False) + "\n")
        write_scalars(writer, report["minibatches"], "benchmark/minibatches")
        writer.flush()

    try:
        writer.add_text("hyperparameters", json.dumps(vars(args), indent=2))
        save()
        spaces = SimpleNamespace(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float64),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32))
        model = base.Agent(spaces).cuda()
        fixtures = make_fixtures(model, args.batch_size)
        for size in args.minibatch_sizes:
            row = report["minibatches"][str(size)] = {}
            benchmark_size(base, model, fixtures, args, size, row, save)
        report["candidate_accepted"] = all(row["adoption"]["accepted"] for row in report["minibatches"].values())
        report["status"] = "complete"
        report["decision"] = "accept" if report["candidate_accepted"] else "reject_no_consistent_measurable_improvement"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        writer.close()
    print(f"RESULT {run_dir / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
