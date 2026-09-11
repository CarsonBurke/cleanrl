"""Paired eager/compiled/CUDA-graph benchmark at production proxy shapes, via mlq.

This measures execution, NOT learning performance. All paths start from identical
complete state and samples. Graph and uncaptured compiled recurrence must agree
exactly. Eager arithmetic drift is reported, not hidden: extreme LR trajectories
can amplify tiny compiler reduction differences. Local fidelity has separate tests.
Rotating execution order limits drift bias; compilation is reported separately.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from cleanrl.plasticity import sample_stream as sample
from cleanrl.shared import runtime


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", default="benchmarks/plasticity/proxy_throughput.json")
    options = parser.parse_args()
    if min(options.batch, options.iterations, options.repeats) <= 0 or options.iterations % 32:
        parser.error("positive counts required and iterations must be divisible by 32")
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    gen = torch.Generator(device="cuda").manual_seed(1)
    a = SimpleNamespace(seeds=1, lr_grid=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
                                       1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                        batch=options.batch, ema=0.999, methods=list(sample.METHODS),
                        switch_at=0.5, huber_k=1.345, var_floor=0.02, nu=5.0,
                        readout_lr=0.002, readout_decay=0.0, cap=20.0)
    n, b = options.iterations, options.batch
    initial = sample.init_mlp(1, len(a.lr_grid), gen, "cuda")
    teacher, direction = sample.teacher(1, gen, "cuda")
    xs = torch.randn(n, 1, b, sample.D_IN, generator=gen, device="cuda")
    clean = sample.teach(teacher, xs.transpose(0, 1).reshape(1, n * b, sample.D_IN))
    clean = clean.reshape(1, n, b, 1).transpose(0, 1).contiguous()
    sigma = (2 * torch.tanh(torch.einsum("si,tsbi->tsb", direction, xs))).exp()
    ys = clean + sigma.unsqueeze(-1) * torch.randn(clean.shape, generator=gen, device="cuda")
    permutations = torch.rand(n, b, generator=gen, device="cuda").argsort(-1)
    states = [sample.make_state(method, initial, 2, b) for method in a.methods]
    counter = torch.zeros(1, dtype=torch.long, device="cuda")
    lr = torch.tensor(a.lr_grid, device="cuda").view(1, -1, 1, 1)
    args = (states, counter, xs, ys, clean, sigma, permutations, lr, a, n // 2)
    compiled, _, chunk, reset, setup = sample.capture_updates(args, 32)
    runs = []
    snapshots = {}
    for repetition in range(options.repeats):
        modes = ("eager", "compiled", "graph")
        order = repetition % len(modes)
        for mode in modes[order:] + modes[:order]:
            reset()
            torch.cuda.synchronize()
            wall, cpu = time.perf_counter(), time.process_time()
            if mode == "eager":
                for _ in range(n):
                    sample.update(*args)
            elif mode == "compiled":
                for _ in range(n):
                    compiled(*args)
            else:
                for _ in range(n // 32):
                    chunk.replay()
            torch.cuda.synchronize()
            elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
            runs.append({"mode": mode, "repetition": repetition, "wall_seconds": elapsed,
                         "cpu_seconds": cpu_elapsed, "stream_samples_per_second": n * b / elapsed,
                         "candidate_samples_per_second": n * b * len(a.methods) * len(a.lr_grid) / elapsed})
            if repetition == 0:
                snapshots[mode] = [t.clone() for t in sample.tensors(states)]
            if int(counter) != n:
                raise AssertionError("replay did not consume the intended sample count")
    max_error = 0.0
    for compiled_state, graph in zip(snapshots["compiled"], snapshots["graph"]):
        torch.testing.assert_close(graph, compiled_state, rtol=0, atol=0, equal_nan=True)
    per_lr_error = torch.zeros(len(a.lr_grid), device="cuda")
    nonfinite_disagreements = 0
    for eager, graph in zip(snapshots["eager"], snapshots["graph"]):
        finite = torch.isfinite(eager) & torch.isfinite(graph)
        same_nonfinite = (torch.isnan(eager) & torch.isnan(graph)) | (eager == graph)
        nonfinite_disagreements += int((~finite & ~same_nonfinite).sum())
        error = torch.where(finite, (eager - graph).abs(), 0)
        max_error = max(max_error, float(error.max()))
        if eager.ndim >= 2 and eager.shape[1] == len(a.lr_grid):
            per_lr_error = torch.maximum(per_lr_error, error.movedim(1, 0).flatten(1).amax(1))
    eager_time = statistics.median(r["wall_seconds"] for r in runs if r["mode"] == "eager")
    graph_time = statistics.median(r["wall_seconds"] for r in runs if r["mode"] == "graph")
    report = {"purpose": "execution benchmark, not a reduced learning experiment", "options": vars(options),
              "config": vars(a), "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "setup_seconds": setup, "runs": runs, "wall_speedup": eager_time / graph_time,
              "compiled_graph_state_parity_max_abs": 0.0,
              "eager_graph_state_drift_max_abs": max_error,
              "eager_graph_state_drift_by_lr": per_lr_error.tolist(),
              "eager_graph_nonfinite_disagreements": nonfinite_disagreements,
              "parity_scope": "compiled versus replay exact, eager versus compiled numerical drift reported"}
    path = Path(options.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
