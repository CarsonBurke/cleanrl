"""Identify finite next-example utility before building predictive plasticity.

Counterfactual probes never modify the replayed SGD carrier. Full existing
streams, first-half-only predictor fitting, explicit nulls and block coupling.
This is a costly identification experiment, not a deployable optimizer.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.runtime import configure_runtime
from cleanrl.plasticity.noisy_stream_diagnostic import Args as SparseArgs, Stream
from cleanrl.plasticity.stock_stream import Args as StockArgs, read_bars, build_stream, build_targets
from cleanrl.plasticity.predictive_transport_benchmark_v2 import save_json
from cleanrl.plasticity.predictive_transport_nonlinear_v2 import (
    INPUTS, HIDDEN, SIZES, forward, draw_inputs,
)
from cleanrl.plasticity.utility_predictor_v1 import identify

CHUNK = 100
REGISTRY = Path("runs/utility_identification_v1_preregistration.json")


@dataclass
class Args:
    task: Literal["sparse", "dense", "alternating", "switch", "stock", "nonlinear"]
    seed: Literal[1] = 1
    bars: str = StockArgs().bars


def probe_actions(device="cuda"):
    joint = torch.tensor([0.125, 0.5, 1.0, 2.0, 8.0], device=device)[:, None].expand(-1, 4)
    singles = torch.eye(4, device=device)
    pairs = torch.stack([singles[i] + singles[j] for i in range(4) for j in range(i + 1, 4)])
    return torch.cat((joint, singles, pairs))


def linear_forward(weight, x):
    return (weight * x).sum(-1)


class Collector:
    """Pure compiled transition; persistent writes occur outside the fused kernel."""
    def __init__(self, initial, rates, nonlinear=False):
        self.nonlinear = nonlinear
        n, p = initial.shape
        self.weight = initial.clone()
        self.previous = initial.clone()
        self.delta = torch.zeros_like(initial)
        self.mean = torch.zeros_like(initial)
        self.square = torch.zeros_like(initial)
        self.t = torch.zeros((), device="cuda")
        self.actions = probe_actions()
        sizes = SIZES if nonlinear else tuple(len(x) for x in torch.tensor_split(torch.arange(p), 4))
        labels = torch.repeat_interleave(torch.arange(4, device="cuda"),
                                        torch.tensor(sizes, device="cuda"))
        masks = (torch.arange(4, device="cuda")[:, None] == labels).float()
        self.sizes = sizes
        self.previous_context = torch.zeros((n, 49), device="cuda")
        self.inputs = (torch.zeros((CHUNK, INPUTS if nonlinear else p), device="cuda"),
                       torch.zeros((CHUNK, n), device="cuda"))
        self.output = (torch.zeros((CHUNK, n, 49), device="cuda"),
                       torch.zeros((CHUNK, n, 15), device="cuda"),
                       torch.zeros((CHUNK, n, 4), device="cuda"),
                       torch.zeros((CHUNK, n), device="cuda"))
        self.states = [self.weight, self.previous, self.delta, self.mean, self.square,
                       self.t, self.previous_context, *self.output]
        predict = forward if nonlinear else linear_forward
        batched = torch.vmap(predict, in_dims=(0, None))

        def loss(w, x, y):
            prediction = predict(w, x)
            return 0.5 * (prediction - y).square(), prediction

        gradients = torch.vmap(torch.func.grad_and_value(loss, has_aux=True), in_dims=(0, None, 0))
        expanded_actions = self.actions[:, labels]

        def step(weight, previous, delta, mean, square, t, x, y):
            if nonlinear:
                gradient, (_, prediction) = gradients(weight, x, y)
            else:
                prediction = batched(weight, x)
                gradient = (prediction - y)[:, None] * x
            norm = gradient.square().sum(1, keepdim=True).sqrt()
            proposal = -gradient / norm.clamp_min(torch.finfo(gradient.dtype).tiny)
            residual = prediction - y
            summaries = [prediction, residual, x.mean().expand(n),
                         x.square().mean().sqrt().expand(n), x.abs().max().expand(n)]
            for w, g, d, m, v in zip(weight.split(sizes, 1), gradient.split(sizes, 1),
                                     proposal.split(sizes, 1), mean.split(sizes, 1), square.split(sizes, 1)):
                summaries.extend((g.mean(1), g.square().mean(1).sqrt(), w.mean(1),
                                  w.square().mean(1).sqrt(), d.square().sum(1).sqrt(),
                                  (g * m).mean(1), v.mean(1), m.square().mean(1).sqrt(),
                                  (d * m).sum(1), (g != 0).float().mean(1),
                                  (d * w).sum(1)))
            context = torch.stack(summaries, 1)
            candidates = previous[:, None, :] + delta[:, None, :] * expanded_actions[None]
            after = batched(candidates.reshape(-1, p), x).reshape(n, 15)
            before = batched(previous, x)
            # Algebraically paired squared-loss differences, accumulated in FP64.
            utility = ((before.double()[:, None] - after.double()) *
                       (0.5 * (before.double()[:, None] + after.double()) - y.double()[:, None])).float()
            if nonlinear:
                endpoint_gradient, _ = gradients(previous + delta, x, y)
            else:
                endpoint_gradient = (batched(previous + delta, x) - y)[:, None] * x
            derivative = -(endpoint_gradient * delta) @ masks.T
            return (context, utility, derivative, prediction,
                    weight - rates[:, None] * gradient,
                    mean + (gradient - mean) / (t + 1),
                    square + (gradient.square() - square) / (t + 1), proposal)

        self.transition = step
        self.kernel = torch.compile(step, fullgraph=True, dynamic=False,
                                    options={"triton.cudagraphs": False})

    def invoke(self, kernel, offset):
        values = kernel(self.weight, self.previous, self.delta, self.mean, self.square,
                        self.t, *(item[offset] for item in self.inputs))
        context, utility, derivative, prediction, weight, mean, square, delta = values
        self.output[0][offset].copy_(self.previous_context)
        for output, value in zip(self.output[1:], (utility, derivative, prediction)):
            output[offset].copy_(value)
        # Separate launches enforce old-parameter retention, avoiding the fused
        # snapshot scheduling failure established in predictive_transport_v1.
        self.previous.copy_(self.weight)
        self.weight.copy_(weight)
        self.mean.copy_(mean)
        self.square.copy_(square)
        self.delta.copy_(delta)
        self.previous_context.copy_(context)
        self.t.add_(1)

    @torch.no_grad()
    def capture(self):
        saved = [v.clone() for v in self.states]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        start = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.invoke(self.kernel, 0)
            stream.synchronize()
            with torch.cuda.graph(self.graph, stream=stream):
                for offset in range(CHUNK):
                    self.invoke(self.kernel, offset)
        finally:
            stream.synchronize()
            for target, source in zip(self.states, saved):
                target.copy_(source)
            torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(self.states, saved))
        return time.perf_counter() - start

    @torch.no_grad()
    def advance(self, x, y):
        length = len(x)
        self.inputs[0][:length].copy_(x)
        self.inputs[1][:length].copy_(y)
        if length == CHUNK:
            self.graph.replay()
        else:
            for offset in range(length):
                self.invoke(self.kernel, offset)
        return tuple(v[:length] for v in self.output)


def frozen_carrier(task):
    research = json.loads(Path("runs/predictive_transport_research_report.json").read_text())
    path = Path(research["valid_experiments"][task]["results"]).with_name("frozen_first_half_selection.json")
    frozen = json.loads(path.read_text())
    rows = frozen["configs"]
    selected = frozen["selection"]
    stream_count = 15 if task == "stock" else 2
    configs = []
    for stream in range(stream_count):
        signal = stream - stream % 5 if task == "stock" else 0
        chosen = next(row for row in rows if row["id"] == selected[f"{signal}:sgd"])
        matching = next(row for row in rows if row["method"] == "sgd" and row["stream"] == stream
                        and row["lr"] == chosen["lr"])
        configs.append(matching)
    return configs, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    configs, selection_source = frozen_carrier(args.task)
    streams = [{key: row[key] for key in ("kind", "alpha")} for row in configs]
    rates = torch.tensor([row["lr"] for row in configs], device="cuda")
    n = len(configs)
    score_after = 2000 if args.task == "stock" else 0
    generator = torch.Generator(device="cuda").manual_seed(1)
    source = features = targets = None
    input_generator = noise_generator = None
    if args.task == "stock":
        source_args = StockArgs(bars=args.bars, seed=1, center=True, steps=0,
                                score_after=2000, null_streams=4, inject_alpha=(0.0, 0.03, 0.1))
        raw_features, raw_targets = build_stream(read_bars(args.bars), source_args)
        rng = np.random.default_rng(1)
        rows, _ = build_targets(raw_targets, raw_features, source_args,
                               [rng.permutation(len(raw_targets)) for _ in range(4)])
        features = torch.cat((torch.as_tensor(raw_features, device="cuda"),
                              torch.ones((len(raw_targets), 1), device="cuda")), 1)
        targets = torch.as_tensor(rows.T.copy(), device="cuda")
        total, p = features.shape
        initial = torch.zeros((n, p), device="cuda")
        del raw_features, raw_targets, rows
    elif args.task == "nonlinear":
        total, p = 50000, sum(SIZES)
        initial_row = torch.cat((torch.randn(INPUTS * HIDDEN, generator=generator, device="cuda") / math.sqrt(INPUTS),
                                 0.1 * torch.randn(HIDDEN, generator=generator, device="cuda"),
                                 torch.randn(HIDDEN, generator=generator, device="cuda") / math.sqrt(HIDDEN),
                                 torch.zeros(1, device="cuda")))
        initial = initial_row.expand(n, -1).clone()
        input_generator = torch.Generator(device="cuda").manual_seed(10001)
        noise_generator = torch.Generator(device="cuda").manual_seed(1000001)
    else:
        total, p = (100000 if args.task == "switch" else 20000), 4096
        source_args = SparseArgs(steps=total, input_dim=p, signal_inputs=p if args.task in ("dense", "alternating") else 1,
                                seeds=1, seed=1, feature_prob=0.01, target_noise_std=math.sqrt(5),
                                spike_prob=0.01, switch_at=0.5 if args.task == "switch" else 0.0,
                                switch_back=0.75 if args.task == "switch" else 0.0, switch_to=2)
        source = Stream(source_args, torch.device("cuda"), False)
        initial = torch.zeros((n, p), device="cuda")
    stamp = f"{time.time():.6f}"
    directory = Path("runs") / f"{args.task}__utility_identification_v1__1__{stamp}"
    directory.mkdir(parents=True)
    names = ("utility_identification_v1.py", "utility_predictor_v1.py", "noisy_stream_diagnostic.py",
             "stock_stream.py", "predictive_transport_nonlinear_v2.py")
    metadata = {"arguments": vars(args), "observations": total, "streams": streams,
                "carrier": "SGD using previously frozen first-half rates; no counterfactual writes",
                "rates": [row["lr"] for row in configs], "frozen_carrier_source": selection_source,
                "registry": str(REGISTRY), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
                "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                "parameters_per_carrier": p, "run_stamp": stamp}
    save_json(directory / "metadata.json", metadata)
    writer = SummaryWriter(str(directory))
    writer.add_text("protocol", json.dumps(metadata, indent=2))
    print(f"run={directory} observations={total} carriers={n}", flush=True)
    collector = Collector(initial, rates, args.task == "nonlinear")
    contexts = torch.empty((total, n, 49), device="cuda")
    utilities = torch.empty((total, n, 15), device="cuda")
    derivatives = torch.empty((total, n, 4), device="cuda")
    error = torch.zeros((2, n), dtype=torch.float64, device="cuda")
    clean_error = torch.zeros_like(error)
    counts = [0, 0]
    midpoint = total // 2
    start = time.perf_counter()
    try:
        with torch.no_grad():
            compile_seconds = collector.capture()
            print(f"compile_capture_seconds={compile_seconds:.3f}; state restored", flush=True)
            observed = 0
            while observed < total:
                boundary = midpoint if observed < midpoint else total
                length = min(CHUNK, boundary - observed)
                clean_rows = None
                if source is not None:
                    inputs, noisy, clean, _, support = source.draw(length, observed + 1)
                    x = inputs[:, 0]
                    noise = noisy - clean
                    if args.task == "alternating":
                        signs = torch.where(torch.arange(p, device="cuda") % 2 == 0, 1.0, -1.0)
                        clean = (x * support * signs).sum(1, keepdim=True)
                        noisy = clean + noise
                    y = torch.cat((noisy, noise), 1)
                    clean_rows = torch.cat((clean, torch.zeros_like(clean)), 1)
                elif args.task == "nonlinear":
                    x = draw_inputs(length, input_generator)
                    clean = x[:, 0] * x[:, 1]
                    noise = math.sqrt(5) * torch.randn(length, generator=noise_generator, device="cuda")
                    y = torch.stack((clean + noise, noise), 1)
                    clean_rows = torch.stack((clean, torch.zeros_like(clean)), 1)
                else:
                    assert features is not None and targets is not None
                    x, y = features[observed:observed + length], targets[observed:observed + length]
                context, utility, derivative, prediction = collector.advance(x, y)
                contexts[observed:observed + length].copy_(context)
                utilities[observed:observed + length].copy_(utility)
                derivatives[observed:observed + length].copy_(derivative)
                phase = int(observed >= midpoint)
                begin = max(0, score_after - observed)
                if begin < length:
                    error[phase].add_((prediction[begin:].double() - y[begin:].double()).square().sum(0))
                    if clean_rows is not None:
                        clean_error[phase].add_((prediction[begin:].double() - clean_rows[begin:].double()).square().sum(0))
                    counts[phase] += length - begin
                observed += length
                if observed % (100000 if args.task == "stock" else 10000) == 0 or observed in (midpoint, total):
                    writer.add_scalar("collection/observations", observed, observed)
                    writer.add_scalar("collection/elapsed_seconds", time.perf_counter() - start, observed)
                    writer.flush()
                    print(f"collection {observed}/{total} seconds={time.perf_counter() - start:.1f}", flush=True)
            torch.cuda.synchronize()
            if not all(bool(torch.isfinite(v).all()) for v in (contexts, utilities, derivatives, collector.weight)):
                raise RuntimeError("Nonfinite carrier/counterfactual records")
            continuity = []
            for index, row in enumerate(configs):
                actual = float(error[0, index] / counts[0])
                expected = row["first_half_mse"]
                continuity.append({"stream": index, "expected_first_half_mse": expected,
                                   "actual_first_half_mse": actual, "absolute_error": abs(actual - expected)})
                if not math.isclose(actual, expected, rel_tol=3e-4, abs_tol=1e-7):
                    raise AssertionError(f"Carrier protocol drift: {continuity[-1]}")
            collection = {"seconds": time.perf_counter() - start, "compile_capture_seconds": compile_seconds,
                          "carrier_mse": (error / error.new_tensor(counts)[:, None]).cpu().tolist(),
                          "carrier_clean_mse": ((clean_error / clean_error.new_tensor(counts)[:, None]).cpu().tolist()
                                                if args.task != "stock" else None),
                          "baseline_continuity": continuity,
                          "counterfactual_actions": collector.actions.cpu().tolist(),
                          "dataset_bytes": sum(v.numel() * v.element_size() for v in (contexts, utilities, derivatives)),
                          "collector_persistent_parameter_state_bytes": sum(v.numel() * v.element_size() for v in
                              (collector.previous, collector.delta, collector.mean, collector.square))}
            save_json(directory / "collection.json", collection)
        # Record at observation t contains context and probe from origin t-1.
        # Discard the initial dummy pair. Final origin has no observed next target.
        result = identify(contexts[1:], utilities[1:], derivatives[1:], collector.actions,
                          midpoint, score_after, streams, writer, directory, save_json, args.task)
        save_json(directory / "results.json", {"metadata": metadata, "collection": collection,
                                               "identification": result, "finished_epoch_seconds": time.time()})
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
