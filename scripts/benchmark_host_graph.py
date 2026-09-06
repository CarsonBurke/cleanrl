"""Host policy mirror: NumPy ufunc chain vs the fused native op graph.

``HostSiTUSphereActor`` is the behavior policy of the live runs. Its forward is
~1.2 MFLOP over 16 rows but is spread across ~85 NumPy calls, so it is bound by
per-call dispatch. ``HostGraphActor`` runs the same graph in one ctypes call.
Both are measured back to back in the same process, on the same parameters and
the same input, because this box normally has training runs on it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.host_actor import (
    HostLReluResActor, HostLReluSphereActor, HostMLP, HostSiTUResActor,
    HostSiTUSphereActor, make_lrelu_res_trunk, make_lrelu_sphere_trunk,
    make_situ_res_trunk, make_situ_sphere_trunk,
)
from cleanrl.shared.host_graph import HostGraphActor
from cleanrl.shared.runtime import configure_runtime


def timeit(fn, iters, warmup=200):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6, min(samples) * 1e6


def compare(label, baseline, candidate, x, iters):
    reference, produced = baseline(x).copy(), candidate(x).copy()
    deviation = np.abs(produced - reference).max()
    base_median, base_min = timeit(lambda: baseline(x), iters)
    graph_median, graph_min = timeit(lambda: candidate(x), iters)
    print(f"{label}")
    print(f"  {type(baseline).__name__:20s} median {base_median:7.2f}us  min {base_min:7.2f}us")
    print(f"  {type(candidate).__name__:20s} median {graph_median:7.2f}us  min {graph_min:7.2f}us")
    print(f"  speedup {base_median / graph_median:5.2f}x median, "
          f"{base_min / graph_min:5.2f}x min; max abs deviation {deviation:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--act-dim", type=int, default=12)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.num_envs, args.obs_dim)).astype(np.float32)

    # Every trunk family host_actor ships, so the fused path is exercised on
    # exactly what the trainers build. The res trunks are two-block by
    # construction and ignore --n-blocks.
    trunks = (
        ("SiTU-sphere", lambda: make_situ_sphere_trunk(args.obs_dim, args.width, args.n_blocks),
         HostSiTUSphereActor, args.n_blocks),
        ("LeakyReluSq-sphere", lambda: make_lrelu_sphere_trunk(args.obs_dim, args.width, args.n_blocks),
         HostLReluSphereActor, args.n_blocks),
        ("SiTU-res", lambda: make_situ_res_trunk(args.obs_dim, args.width), HostSiTUResActor, 2),
        ("LeakyReluSq-res", lambda: make_lrelu_res_trunk(args.obs_dim, args.width),
         HostLReluResActor, 2),
    )
    for label, make_trunk, mirror_cls, blocks in trunks:
        seq = nn.Sequential(make_trunk(), nn.Linear(args.width, args.act_dim)).cuda()
        compare(
            f"{label} trunk: {args.num_envs} rows, width {args.width}, "
            f"{blocks} blocks, {args.obs_dim}->{args.act_dim}",
            mirror_cls(seq, args.num_envs),
            HostGraphActor(seq, args.num_envs),
            x, args.iters,
        )

    mlp = nn.Sequential(
        nn.Linear(args.obs_dim, args.width), nn.Tanh(),
        nn.Linear(args.width, args.width), nn.Tanh(),
        nn.Linear(args.width, args.act_dim),
    ).cuda()
    compare(
        f"Plain tanh MLP: {args.num_envs} rows, width {args.width}, "
        f"{args.obs_dim}->{args.act_dim}",
        HostMLP(mlp, args.num_envs),
        HostGraphActor(mlp, args.num_envs),
        x, args.iters,
    )


if __name__ == "__main__":
    main()
