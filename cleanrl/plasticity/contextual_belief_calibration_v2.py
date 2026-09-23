"""Full20k fixed-prediction acquisition/calibration/revision component experiment."""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity.contextual_belief_v2 import CONTROLS, ContextualBeliefState
from cleanrl.plasticity.predictive_transport_benchmark_v2 import save_json
from cleanrl.shared.runtime import configure_runtime

CHUNK = 100
TOTAL = 20000


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    rng = torch.Generator(device="cuda").manual_seed(1)
    states = [ContextualBeliefState(torch.zeros((1, 1), device="cuda"), control=name) for name in CONTROLS]
    inputs = (torch.zeros((CHUNK, 1, 1, 2), device="cuda"), torch.zeros((CHUNK, 1), device="cuda"))
    outputs = torch.zeros((CHUNK, len(states), 7), dtype=torch.float64, device="cuda")
    active = torch.ones((1, 1), dtype=torch.bool, device="cuda")

    def transition(context, residual):
        values, updates = [], []
        for state in states:
            p, updated, metrics = state.belief_transition(context, residual, active)
            values.append(torch.stack((p["prediction"][0, 0], p["noise"][0, 0],
                                       p["predictive_variance"][0, 0], p["gain"][0, 0],
                                       metrics[0][0, 0], metrics[3][0, 0], metrics[4][0, 0])))
            updates.append(updated)
        return torch.stack(values), updates

    kernel = torch.compile(transition, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})

    def invoke(offset):
        values, updates = kernel(*(value[offset] for value in inputs))
        outputs[offset].copy_(values)
        for state, updated in zip(states, updates):
            for name, value in updated.items():
                getattr(state, name).copy_(value)

    buffers = [value for state in states for value in state.buffers().values()]
    saved = [value.clone() for value in buffers]
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
        for value, original in zip(buffers, saved):
            value.copy_(original)
        torch.cuda.synchronize()
    assert all(torch.equal(value, original) for value, original in zip(buffers, saved))
    directory = Path("runs") / f"calibration__contextual_belief_v2__1__{time.time():.6f}"
    directory.mkdir(parents=True)
    writer = SummaryWriter(str(directory))
    sums = torch.zeros((3, 2, len(states), 8), dtype=torch.float64, device="cuda")
    counts = torch.zeros((3, 2), dtype=torch.float64, device="cuda")
    metadata = {"seed": 1, "observations": TOTAL, "scope": "Fixed-prediction inference experiment; not network-learning evidence",
                "phases": ["0:5000 mean2*c", "5000:15000 mean0", "15000:20000 mean-2*c"],
                "evaluation": "Last2500 observations of each phase; inference runs throughout all20000",
                "variance": {"context_-1": 0.25, "context_+1": 9},
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("contextual_belief_v2.py", "contextual_belief_calibration_v2.py")},
                "preregistration_sha256": hashlib.sha256(Path("runs/contextual_belief_v2_preregistration.json").read_bytes()).hexdigest()}
    writer.add_text("protocol", json.dumps(metadata, indent=2))
    print(f"run={directory} observations={TOTAL}", flush=True)
    try:
        for observed in range(0, TOTAL, CHUNK):
            context = 2 * (torch.rand(CHUNK, generator=rng, device="cuda") < 0.5).float() - 1
            phase = 0 if observed < 5000 else (1 if observed < 15000 else 2)
            mean = (2 if phase == 0 else (0 if phase == 1 else -2)) * context
            noise = torch.where(context < 0, 0.5, 3.0) * torch.randn(CHUNK, generator=rng, device="cuda")
            inputs[0][..., 0].copy_(context[:, None, None])
            inputs[0][..., 1].zero_()
            inputs[1].copy_((mean + noise)[:, None])
            graph.replay()
            if observed >= (2500, 12500, 17500)[phase]:
                values = torch.cat((outputs, (outputs[:, :, 0] - mean.double()[:, None]).square()[..., None]), -1)
                for index, mask in enumerate((context < 0, context > 0)):
                    sums[phase, index].add_(torch.where(mask[:, None, None], values, 0.0).sum(0))
                    counts[phase, index].add_(mask.sum())
            if (observed + CHUNK) % 1000 == 0:
                for index, name in enumerate(CONTROLS):
                    writer.add_scalar(f"calibration/{name}/nll", outputs[:, index, 4].mean(), observed + CHUNK)
                    writer.add_scalar(f"calibration/{name}/gain", outputs[:, index, 3].mean(), observed + CHUNK)
                writer.flush()
        averages = sums / counts[:, :, None, None]
        if not bool(torch.isfinite(averages).all()):
            raise RuntimeError("Nonfinite component calibration")
        rows = []
        fields = ("predicted_mean", "noise_variance", "predictive_variance", "gain", "nll",
                  "bias_probability", "change_probability", "mean_prediction_mse")
        for method, name in enumerate(CONTROLS):
            rows.append({"method": name, "phases": [
                {"phase": label, "states": [{"context": context, "count": int(counts[phase, index]),
                   **dict(zip(fields, averages[phase, index, method].cpu().tolist()))}
                   for index, context in enumerate((-1, 1))],
                 "nll": float(sums[phase, :, method, 4].sum() / counts[phase].sum()),
                 "mean_prediction_mse": float(sums[phase, :, method, 7].sum() / counts[phase].sum())}
                for phase, label in enumerate(("acquisition", "calibrated", "revision"))]})
        save_json(directory / "results.json", {"metadata": metadata, "results": rows,
                                              "seconds": time.perf_counter() - started})
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
