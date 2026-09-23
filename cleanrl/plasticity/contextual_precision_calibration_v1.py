"""Full 20k prequential component experiment, not network-learning evidence.

Fixed prediction; recurring target-free context +/-1; true residual mean 2*c;
noise variance .25 or9. All controls receive identical observations. Only
optimizer calibration coefficients learn. No model layer or readout is added.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.runtime import configure_runtime
from cleanrl.plasticity.contextual_precision_v1 import ContextualPrecisionState
from cleanrl.plasticity.predictive_transport_benchmark_v2 import save_json

CHUNK = 100
TOTAL = 20000
METHODS = ("conditional", "unconditional", "uncentered_noise")


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    rng = torch.Generator(device="cuda").manual_seed(1)
    weight = torch.zeros((1, 1), device="cuda")
    states = [ContextualPrecisionState(weight, control=name) for name in METHODS]
    context_buffer = torch.zeros((CHUNK, 1, 1), device="cuda")
    residual_buffer = torch.zeros((CHUNK, 1), device="cuda")
    active = torch.ones((1, 1), dtype=torch.bool, device="cuda")
    # Per-step, per-control: pre-observation mean, variance and Gaussian NLL.
    outputs = torch.zeros((CHUNK, len(METHODS), 3), device="cuda")
    totals = torch.zeros((2, len(METHODS), 4), dtype=torch.float64, device="cuda")
    counts = torch.zeros(2, dtype=torch.float64, device="cuda")

    def step(context, residual):
        rows = []
        updates = []
        for state in states:
            _, mean, _, variance = state.predictive(context)
            _, updated, stats = state.calibration_transition(context, residual, active)
            rows.append(torch.stack((mean[0, 0], variance[0, 0], stats[0][0].float())))
            updates.append(updated)
        return torch.stack(rows), updates

    compiled = torch.compile(step, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})

    def invoke(offset):
        values, updates = compiled(context_buffer[offset], residual_buffer[offset])
        outputs[offset].copy_(values)
        for state, updated in zip(states, updates):
            for name, value in updated.items():
                getattr(state, name).copy_(value)

    buffers = [value for state in states for value in state.buffers().values()]
    snapshots = [value.clone() for value in buffers]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                invoke(0)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for offset in range(CHUNK):
                invoke(offset)
    finally:
        stream.synchronize()
        for value, saved in zip(buffers, snapshots):
            value.copy_(saved)
        torch.cuda.synchronize()
    assert all(torch.equal(value, saved) for value, saved in zip(buffers, snapshots))
    directory = Path("runs") / f"calibration__contextual_precision_v1__1__{time.time():.6f}"
    directory.mkdir(parents=True)
    writer = SummaryWriter(str(directory))
    sources = ("contextual_precision_v1.py", "contextual_precision_calibration_v1.py")
    metadata = {"seed": 1, "observations": TOTAL, "evaluation_start": 15000,
                "methods": METHODS, "mean": "2*context", "variance": {"-1": 0.25, "+1": 9.0},
                "scope": "Fixed-network conditional Gaussian calibration only; not model-learning evidence",
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
                "registry_sha256": hashlib.sha256(Path("runs/contextual_precision_v1_preregistration.json").read_bytes()).hexdigest()}
    writer.add_text("protocol", json.dumps(metadata, indent=2))
    print(f"run={directory} observations={TOTAL}", flush=True)
    try:
        for observed in range(0, TOTAL, CHUNK):
            context = 2 * (torch.rand((CHUNK,), device="cuda", generator=rng) < 0.5).float() - 1
            variance = torch.where(context < 0, 0.25, 9.0)
            mean = 2 * context
            residual = mean + variance.sqrt() * torch.randn(CHUNK, device="cuda", generator=rng)
            context_buffer.copy_(context[:, None, None])
            residual_buffer.copy_(residual[:, None])
            graph.replay()
            if observed >= 15000:
                for regime, mask in enumerate((context < 0, context > 0)):
                    values = torch.stack((outputs[:, :, 0].double(), outputs[:, :, 1].double(),
                                          outputs[:, :, 2].double(),
                                          (outputs[:, :, 0].double() - mean.double()[:, None]).square()), -1)
                    totals[regime].add_(torch.where(mask[:, None, None], values, 0.0).sum(0))
                    counts[regime].add_(mask.sum())
            if (observed + CHUNK) % 5000 == 0:
                for index, name in enumerate(METHODS):
                    writer.add_scalar(f"calibration/{name}/chunk_nll", outputs[:, index, 2].mean(), observed + CHUNK)
                writer.flush()
                print(f"progress {observed + CHUNK}/{TOTAL}", flush=True)
        average = totals / counts[:, None, None]
        if not bool(torch.isfinite(average).all()):
            raise RuntimeError("Nonfinite calibration experiment")
        rows = []
        for index, name in enumerate(METHODS):
            rows.append({"method": name, "regimes": [
                {"context": context, "count": int(counts[regime]),
                 "predicted_mean": float(average[regime, index, 0]),
                 "predicted_variance": float(average[regime, index, 1]),
                 "gaussian_nll": float(average[regime, index, 2]),
                 "mean_estimate_mse": float(average[regime, index, 3])}
                for regime, context in enumerate((-1, 1))],
                "overall_nll": float(totals[:, index, 2].sum() / counts.sum())})
        result = {"metadata": metadata, "results": rows, "seconds": time.perf_counter() - started,
                  "finished_epoch_seconds": time.time()}
        save_json(directory / "results.json", result)
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
