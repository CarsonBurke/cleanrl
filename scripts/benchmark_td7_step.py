"""Time one TD7-family train step (CUDA-graph replay) and the host acting path.

Loads a trainer file by path, builds its ``TD7Agent`` on synthetic replay data
and reports:
  - train-step wall time per replay (actor/critic graphs alternate as in training),
  - the eager (uncaptured) step for attribution,
  - host policy latency for one row, and its max abs deviation from the device policy.

Usage (queue it through mlq; it uses the GPU):
  .venv/bin/python scripts/benchmark_td7_step.py cleanrl/td7/fast/td7_fast_sdnoise_v1.py [--steps 2000]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path):
    spec = importlib.util.spec_from_file_location("td7_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trainer")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--state-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=6)
    cli = parser.parse_args()

    module = load(cli.trainer)
    module.configure_runtime()
    args = module.Args(compile=not cli.no_compile)
    device = torch.device("cuda")
    writer = _NullWriter()
    t0 = time.perf_counter()
    agent = module.TD7Agent(cli.state_dim, cli.action_dim, 1000, args, device, writer)
    torch.cuda.synchronize()
    print(f"build+capture: {time.perf_counter() - t0:.1f}s")

    # Synthetic replay: 200k random transitions with max priority.
    n = 200_000
    with torch.no_grad():
        agent.storage[:n].normal_()
        agent.storage[:n, -1] = 1.0
        agent.priority[:n] = 1.0
    agent.size = n

    for _ in range(50):
        agent.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cli.steps):
        agent.train()
    torch.cuda.synchronize()
    graph_us = (time.perf_counter() - t0) / cli.steps * 1e6
    print(f"graph train step: {graph_us:.0f} us  ({1e6 / graph_us:.0f} steps/s)")

    eager = [agent._make_step(update_actor=False), agent._make_step(update_actor=True)]
    for i in range(20):
        eager[i % 2]()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = max(cli.steps // 10, 50)
    for i in range(steps):
        eager[i % 2]()
    torch.cuda.synchronize()
    print(f"eager step (same kernels, no graph): {(time.perf_counter() - t0) / steps * 1e6:.0f} us")

    agent.policy.refresh()
    state = np.random.default_rng(0).standard_normal((1, cli.state_dim)).astype(np.float32)
    for _ in range(100):
        agent.policy(state)
    t0 = time.perf_counter()
    for _ in range(2000):
        agent.policy(state)
    print(f"host policy (1 row): {(time.perf_counter() - t0) / 2000 * 1e6:.1f} us")
    states = np.random.default_rng(1).standard_normal((64, cli.state_dim)).astype(np.float32)
    host = agent.policy(states)
    with torch.no_grad():
        x = torch.from_numpy(states).to(device)
        dev = agent.actor(x, agent.fixed_encoder.zs(x))
    for name, h, d in zip(("mu", "log_std"), host, dev):
        if h is not None:
            print(f"host-vs-device max |{name}| error: {np.abs(h - d.cpu().numpy()).max():.2e}")


class _NullWriter:
    def add_scalar(self, *a, **k):
        pass


if __name__ == "__main__":
    main()
