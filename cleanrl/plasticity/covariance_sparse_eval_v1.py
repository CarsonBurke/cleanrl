"""Exact LINEAR specialization of network_bayes_stream_v2, not its neural result.

The 4096-input width-64 MLP posterior is roughly 280 GB; this diagnostic instead
conditions the full 4096 x 4096 FP32 posterior of one bias-free linear block.
The frozen v2 update is reused verbatim, including fan-in-normalized prior and
process noise. Residual-noise estimation is causal, not supplied by the teacher.
All reported online predictions precede their label updates. Exact Bernoulli
risk uses teacher support only outside the optimizer, without held-out tensors.

The 60000-observation horizon is a ceiling: automatic resource culling is ON by
default (--no-autocull explicitly opts out). At coherent training checkpoints,
every candidate is monitored using interval noisy MSE / target energy and exact
clean risk, each with absolute min_delta=1e-4. Either improving signal protects
a candidate; every candidate must stagnate before a plateau stop. The policy
uses an 8192-observation EMA half-life, 16384-observation phase warmup, at least
4096 observations between evaluations, and patience three. The exact switch
checkpoint is excluded because its interval and frozen risk use different
teachers; the first post-switch checkpoint starts a fresh phase at the switch.
Pruned runs retain partial artifacts and exit 75, not success; queue with
--max-attempts 1 so resource stops do not retry or release success dependencies.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import network_bayes_stream_v2 as v2
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy


@dataclass
class Args:
    seed: int = 1
    steps: int = 60_000
    input_dim: int = 4096
    feature_prob: float = 0.01
    noise_sigma: float = 1.0
    adam_lrs: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    softhinge_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    noise_rate: float = 0.001
    veto_z: float = 4.0
    hinge_sharpness: float = 24.0
    graph_steps: int = 100
    log_every: int = 1000
    autocull: bool = True
    output: str = "runs"


def validate_args(a):
    if a.seed != 1 or a.steps != 60_000 or a.input_dim != 4096:
        raise ValueError("research protocol requires seed1, 60000 observations, 4096 inputs")
    if a.feature_prob != 0.01 or a.noise_sigma != 1.0:
        raise ValueError("research protocol requires Bernoulli(.01) inputs and Gaussian sigma1")
    if min(a.graph_steps, a.log_every) <= 0 or a.graph_steps > a.steps:
        raise ValueError("positive graph/log cadences required; graph cannot exceed stream")
    if not 0 <= a.noise_rate <= 1 or a.veto_z <= 0 or a.hinge_sharpness <= 0:
        raise ValueError("invalid noise estimator or soft-hinge configuration")
    for grid in (a.adam_lrs, a.softhinge_lrs):
        if not grid or list(grid) != sorted(set(grid)) or any(not math.isfinite(x) or x <= 0 for x in grid):
            raise ValueError("learning-rate grids must be finite, positive and strictly increasing")


@torch.no_grad()
def draw_stream(a, device):
    """One shared feature/noise stream, with independent generators for each.

    Boolean storage is 246 MB at full horizon. Generation is chunked, and no
    large float feature tensor or sampled held-out evaluation set is retained.
    """
    feature_rng = torch.Generator(device=device).manual_seed(a.seed)
    noise_rng = torch.Generator(device=device).manual_seed(a.seed + 1_000_003)
    xs = torch.empty((a.steps, a.input_dim), dtype=torch.bool, device=device)
    for start in range(0, a.steps, 1000):
        end = min(start + 1000, a.steps)
        xs[start:end].copy_(torch.rand((end - start, a.input_dim), device=device,
                                      generator=feature_rng) < a.feature_prob)
    noise = torch.randn(a.steps, device=device, generator=noise_rng) * a.noise_sigma
    return xs, noise


def teacher_labels(xs, noise, view):
    """Zero-based support moves from coordinate 0 to 1 after exactly N/2 labels."""
    if view == "null":
        clean = torch.zeros_like(noise)
    elif view == "stationary":
        clean = xs[:, 0].to(noise.dtype)
    elif view == "change":
        clean = torch.cat((xs[:len(xs) // 2, 0], xs[len(xs) // 2:, 1])).to(noise.dtype)
    else:
        raise ValueError(f"unknown view {view!r}")
    return clean + noise, clean


def bernoulli_power(weight, probability):
    w = weight.double()
    return (w.square() * probability * (1 - probability)).sum(-1) + (w * probability).sum(-1).square()


def exact_risk(weight, support, probability, stale_index=None):
    """Exact clean risk; stale leakage is a SUBSET of distractor leakage.

    Uncentered Bernoulli features require the signed mean cross term. Reporting
    stale power alone is not an additive partition: its mean can cancel other
    distractors, so stale/rest powers and their signed cross are also returned.
    """
    w, target = weight.double(), support.double()
    active = target != 0
    signal_error = (w - target) * active
    junk = w * (~active)
    junk_mean = (junk * probability).sum(-1)
    junk_variance = (junk.square() * probability * (1 - probability)).sum(-1)
    stale = torch.zeros_like(junk)
    if stale_index is not None:
        stale[..., stale_index] = junk[..., stale_index]
    rest = junk - stale
    stale_mean = (stale * probability).sum(-1)
    rest_mean = (rest * probability).sum(-1)
    return {
        "clean_mse": bernoulli_power(w - target, probability),
        "zero_clean_mse": bernoulli_power(target, probability).expand(w.shape[0]),
        "prediction_energy": bernoulli_power(w, probability),
        "signal_reconstruction_mse": bernoulli_power(signal_error, probability),
        "distractor_variance": junk_variance,
        "distractor_mean": junk_mean,
        "distractor_mean_squared": junk_mean.square(),
        "distractor_leakage_mse": junk_variance + junk_mean.square(),
        "signal_distractor_cross": 2 * (signal_error * probability).sum(-1) * junk_mean,
        "stale_support_leakage_mse": bernoulli_power(stale, probability),
        "stale_support_mean": stale_mean,
        "remaining_distractor_leakage_mse": bernoulli_power(rest, probability),
        "stale_remaining_cross": 2 * stale_mean * rest_mean,
    }


class LinearLearner(v2.Learner):
    """Small learner API: only x and noisy y reach update().

    `network_update`, `snapshot`, and `restore` are the frozen v2 methods. The
    output sensitivity is identically one; no neural approximation is involved.
    With noise_rate=0 and diffusion=0 this is ordinary Gaussian conditioning.
    """

    def __init__(self, method, grid, a, xs, ys, *, diffusion=1e-5):
        if method not in {"network", "adam", "softhinge"}:
            raise ValueError(f"unknown method {method!r}")
        self.method, self.a = method, a
        self.xs, self.ys = xs, ys
        k, d, device = len(grid), xs.shape[1], xs.device
        self.weights = [torch.zeros((k, 1, d), device=device)]
        self.scale = torch.tensor(grid, device=device)
        self.index = torch.zeros((), device=device, dtype=torch.int64)
        self.steps = torch.zeros((), device=device)
        self.noise = torch.ones(k, device=device)
        self.prediction = torch.zeros(k, device=device)
        # Columns: y², prediction², y*prediction, (y-prediction)². Double
        # accumulation avoids a 60k-sample FP32 sum obscuring a weak signal.
        self.moments = torch.zeros((k, 4), device=device, dtype=torch.float64)
        self.mutable = [*self.weights, self.index, self.steps, self.noise,
                        self.prediction, self.moments]
        self.capture_steps = 1 if method == "network" else min(a.graph_steps, len(xs))
        if method == "network":
            prior = (self.scale / d)[:, None].expand(k, d)
            self.cov = torch.zeros((k, d, d), device=device, dtype=torch.float32)
            self.cov.diagonal(dim1=-2, dim2=-1).copy_(prior)
            self.process = diffusion * prior
            self.mutable.append(self.cov)
        else:
            self.m = torch.zeros((k, d), device=device)
            self.v = torch.zeros_like(self.m)
            self.mutable += [self.m, self.v]
            if method == "softhinge":
                self.running_sum = torch.zeros_like(self.m)
                self.running_sq = torch.zeros_like(self.m)
                self.mutable += [self.running_sum, self.running_sq]

    @torch.no_grad()
    def update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0).float()
        y = self.ys.index_select(0, ix).squeeze(0)
        w = self.weights[0][:, 0, :]
        prediction = (w * x).sum(-1)
        self.prediction.copy_(prediction)
        # Record before ANY parameter update: current y only scores this forecast.
        residual = prediction - y
        self.moments.add_(torch.stack((y.square().expand_as(prediction), prediction.square(),
                                       y * prediction, residual.square()), -1).double())
        self.steps.add_(1)
        if self.method == "network":
            inputs = (x.unsqueeze(0).expand(w.shape[0], -1),)
            sensitivities = (torch.ones_like(prediction).unsqueeze(-1),)
            self.network_update(inputs, sensitivities, residual, self.noise)
            self.noise.lerp_(residual.square(), self.a.noise_rate)
        else:
            grad = residual.unsqueeze(-1) * x
            self.m.lerp_(grad, 0.1)
            self.v.lerp_(grad.square(), 0.001)
            update = self.scale[:, None] * (self.m / (1 - .9 ** self.steps)) / (
                (self.v / (1 - .999 ** self.steps)).sqrt() + 1e-5)
            if self.method == "softhinge":
                # Corrected diagnostic formula, fixed z=4: gate uses PAST signed
                # gradients. Current residual cannot make its own gate permissive.
                # Fixed-z softhinge has no randomized twin, forgetting or amplifier.
                t_sq = self.running_sum.square() / self.running_sq.clamp_min(1e-30)
                raw = 1 - self.a.veto_z ** 2 / t_sq.clamp_min(1e-30)
                certainty = F.softplus(self.a.hinge_sharpness * raw) / self.a.hinge_sharpness
                update = update * certainty
                self.running_sum.add_(grad)
                self.running_sq.add_(grad.square())
            w.sub_(update)
        self.index.add_(1)

    @torch.no_grad()
    def capture(self):
        """Compile once, graph full blocks and exact single-step tails, reset ALL state."""
        initial = self.snapshot()
        compiled = torch.compile(self.update, fullgraph=True, mode="max-autotune-no-cudagraphs")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        try:
            with torch.cuda.stream(stream):
                compiled()
                self.restore(initial)
                compiled()
            torch.cuda.current_stream().wait_stream(stream)
            self.restore(initial)
            torch.cuda.synchronize()
            for count in sorted({1, self.capture_steps}):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(count):
                        compiled()
                stream.synchronize()
                self.restore(initial)
                torch.cuda.synchronize()
                graphs[count] = graph
        finally:
            stream.synchronize()
            self.restore(initial)
            torch.cuda.synchronize()
        return graphs


def moment_report(total, count):
    values = (total / count).cpu().tolist()
    names = ("target_energy", "prediction_energy", "target_prediction", "noisy_mse")
    report = {name: [row[i] for row in values] for i, name in enumerate(names)}
    report["prediction_over_target_energy"] = [row[1] / row[0] if row[0] > 0 else None for row in values]
    return report


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(v2.finite_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def run_arm(a, method, grid, view, xs, noise, root, writer, *, diffusion=1e-5, selected=None):
    """Train all candidates on the stationary view; transfer ONLY its locked LR."""
    name = f"covariance_q{diffusion:g}" if method == "network" else method
    labels, _ = teacher_labels(xs, noise, view)
    torch.cuda.reset_peak_memory_stats()
    learner = LinearLearner(method, grid, a, xs, labels, diffusion=diffusion)
    torch.cuda.synchronize()
    started = time.perf_counter()
    graphs = learner.capture()
    startup = time.perf_counter() - started
    prefix, switch = a.steps // 4, a.steps // 2
    endpoints = sorted(set(range(a.log_every, a.steps + 1, a.log_every)) | {prefix, switch, a.steps})
    previous = torch.zeros_like(learner.moments)
    prefix_moments = None
    previous_step = 0
    curves, checkpoint_weights = [], []
    selection = selected
    cull = ProxyCull(len(grid), {"normalized_noisy_mse": 1e-4, "clean_mse": 1e-4}) if a.autocull else None
    prune_record = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in endpoints:
        count = step - previous_step
        blocks, tail = divmod(count, learner.capture_steps)
        for _ in range(blocks):
            graphs[learner.capture_steps].replay()
        for _ in range(tail):
            graphs[1].replay()
        total = learner.moments.clone()
        interval = moment_report(total - previous, count)
        # At the switch endpoint, evaluate the new teacher BEFORE its first label:
        # this makes instant stale leakage visible rather than hiding it for 1k steps.
        moved = view == "change" and step >= switch
        support = torch.zeros(a.input_dim, device=xs.device)
        if view != "null":
            support[1 if moved else 0] = 1
        weights = learner.weights[0][:, 0, :]
        risk = {key: value.cpu().tolist() for key, value in exact_risk(
            weights, support, a.feature_prob, stale_index=0 if moved else None).items()}
        row = {"step": step, "interval_start_exclusive": previous_step,
               "teacher_for_frozen_risk": "new" if moved else view,
               "interval_prequential": interval,
               "cumulative_prequential": moment_report(total, step), "exact_risk": risk,
               "coefficient_0": weights[:, 0].cpu().tolist(),
               "coefficient_1": weights[:, 1].cpu().tolist()}
        if cull is not None and not (view == "change" and step == switch):
            normalized_mse = [
                mse / energy if math.isfinite(energy) and energy > 0 else float("nan")
                for mse, energy in zip(interval["noisy_mse"], interval["target_energy"])
            ]
            prune_record = cull.observe(
                step, {"normalized_noisy_mse": normalized_mse, "clean_mse": risk["clean_mse"]},
                phase="post_switch" if moved else view, phase_start=switch if moved else 0,
            )
        row["autocull_state"] = cull.state_dict() if cull is not None else None
        if step == prefix:
            prefix_moments = total.clone()
            if selection is None and prune_record is None:
                scores = row["cumulative_prequential"]["noisy_mse"]
                finite = [i for i, score in enumerate(scores) if math.isfinite(score)]
                if not finite:
                    raise RuntimeError(f"all {name} candidates nonfinite on stationary prefix")
                chosen = min(finite, key=scores.__getitem__) if method != "network" else 0
                selection = {"view": "stationary", "endpoint": prefix, "grid": list(grid),
                             "chosen_index": chosen, "chosen": grid[chosen],
                             "criterion": "fixed_original_prior" if method == "network" else "prefix_mean_prequential_noisy_mse",
                             "candidate_scores": scores, "candidate_endpoint_metrics": row,
                             "boundary_choice": method != "network" and chosen in (0, len(grid) - 1)}
                # Persist choice before any suffix observation is processed/reported.
                save_json(root / f"selection_{name}.json", selection)
        if step > prefix:
            row["suffix_prequential"] = moment_report(total - prefix_moments, step - prefix)
        curves.append(row)
        checkpoint_weights.append(weights.cpu().clone())
        for index, candidate in enumerate(grid):
            tag = f"{view}/{name}/candidates/{candidate:g}"
            writer.add_scalar(f"{tag}/noisy_mse", interval["noisy_mse"][index], step)
            writer.add_scalar(f"{tag}/clean_mse", risk["clean_mse"][index], step)
        local_chosen = selection["chosen_index"] if view == "stationary" and selection is not None else 0
        if selection is not None:
            for section in ("interval_prequential", "exact_risk"):
                for metric, values in row[section].items():
                    value = values[local_chosen]
                    if value is not None:
                        writer.add_scalar(f"{view}/{name}/{section}/{metric}", value, step)
        previous, previous_step = total, step
        if prune_record is not None:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Compact parameter checkpoints preserve future probes without dense P snapshots.
    checkpoint_path = root / f"checkpoints_{view}_{name}.pt"
    status = "pruned" if prune_record is not None else "complete"
    consumed = int(learner.index.item())
    policy_state = cull.state_dict() if cull is not None else None
    checkpoint_temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save({"steps": [row["step"] for row in curves], "grid": list(grid),
                "weights": torch.stack(checkpoint_weights),
                "latest_prequential_prediction": learner.prediction.cpu(),
                "processed_observations": consumed, "status": status,
                "autocull_state": policy_state, "selection": selection}, checkpoint_temporary)
    checkpoint_temporary.replace(checkpoint_path)
    chosen = (selection["chosen_index"] if view == "stationary" else 0) if selection is not None else None
    result = {"method": name, "view": view, "grid": list(grid), "selection": selection,
              "status": status, "planned_observations": a.steps,
              "autocull": a.autocull, "autocull_state": policy_state, "prune": prune_record,
              "local_chosen_index": chosen, "selection_transferred_unchanged": view != "stationary",
              "prior_scale": 1.0 if method == "network" else None,
              "initial_coordinate_variance": 1 / a.input_dim if method == "network" else None,
              "diffusion_fraction": diffusion if method == "network" else None,
              "noise_estimator": "previous residual-square EMA" if method == "network" else None,
              "noise_rate": a.noise_rate if method == "network" else None,
              "startup_seconds": startup, "training_and_reporting_seconds": elapsed,
              "observations_per_second": consumed / elapsed,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "processed_observations": consumed,
              "checkpoint_path": str(checkpoint_path), "curves": curves}
    save_json(root / f"{view}_{name}.json", result)
    del graphs, learner
    if prune_record is not None:
        if selection is None:
            save_json(root / f"selection_{name}.json", None)
        prune_proxy(root, f"{view}/{name}", prune_record)
    return result, selection


def main():
    a = tyro.cli(Args)
    validate_args(a)
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU or eager learner fallback")
    device = torch.device("cuda")
    root = Path(a.output) / f"SparseStream__covariance_linear_v1__{a.seed}__{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    xs, noise = draw_stream(a, device)
    result = {
        "args": asdict(a), "run_dir": str(root),
        "status": "running",
        "scope": "exact bias-free LINEAR specialization of v2 full posterior; NOT neural stock evaluation",
        "protocol": {"seed_count": 1, "selection_endpoint": a.steps // 4,
                     "switch_after_observations": a.steps // 2,
                     "signal_coefficient": 1, "initial_support": 0, "moved_support": 1,
                     "initial_mean": "all coefficients zero; no bias coordinate",
                     "null_teacher": "clean=0; the SAME independent Gaussian label noise remains",
                     "pairing": "identical features and additive Gaussian noise across every arm/view",
                     "control_rng": "none consumed: fixed-z softhinge has no randomized twin; learner never draws from data RNGs",
                     "selection": "first-quarter stationary prequential noisy MSE; no suffix/null/change tuning",
                     "covariance": "fixed prior_scale1, P0=I/4096; Q=diffusion*P0; no prior-grid selection",
                     "risk": "exact independent uncentered Bernoulli moments; support never reaches learner",
                     "switch_checkpoint": "frozen risk is against NEW support at step30000; online interval still old",
                     "autocull": "default-on training normalized noisy MSE and exact clean risk; all candidates must plateau; switch endpoint excluded; phase resets at switch",
                     "precision": "FP32 full posterior, TF32 off; FP64 moment accumulation and exact-risk evaluation",
                     "interpretation": "single-seed paired diagnostic, not statistical confidence or global optimality"},
        "covariance_bytes_per_arm": a.input_dim ** 2 * 4,
        "arms": {},
    }
    writer = SummaryWriter(str(root))
    writer.add_text("protocol", json.dumps(result["protocol"], indent=2))
    writer.add_text("hyperparameters", json.dumps(asdict(a), indent=2))
    exit_code = 0
    try:
        for method, grid, diffusion in (("network", (1.0,), 1e-5), ("network", (1.0,), 0.0),
                                        ("adam", a.adam_lrs, 0.0), ("softhinge", a.softhinge_lrs, 0.0)):
            selected = None
            for view in ("stationary", "null", "change"):
                current_grid = grid if selected is None else (selected["chosen"],)
                name = f"covariance_q{diffusion:g}" if method == "network" else method
                row, selected = run_arm(a, method, current_grid, view, xs, noise, root, writer,
                                        diffusion=diffusion, selected=selected)
                result["arms"][f"{view}/{row['method']}"] = row
                result["elapsed_seconds"] = time.perf_counter() - started
                save_json(root / "results.json", result)
                print(json.dumps({"view": view, "method": row["method"], "chosen": selected["chosen"],
                                  "observations": row["processed_observations"],
                                  "seconds": row["training_and_reporting_seconds"]}), flush=True)
        result["status"] = "complete"
        save_json(root / "results.json", result)
    except ProxyPruned as error:
        result["status"] = "pruned"
        result["prune"] = error.record
        arm_key = error.record["arm"]
        artifact_name = arm_key.replace("/", "_")
        result["arms"][arm_key] = json.loads((root / f"{artifact_name}.json").read_text())
        result["elapsed_seconds"] = time.perf_counter() - started
        save_json(root / "results.json", result)
        exit_code = PRUNED_EXIT_CODE
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["elapsed_seconds"] = time.perf_counter() - started
        save_json(root / "results.json", result)
        raise
    finally:
        writer.close()
    print(f"RESULTS {root / 'results.json'}", flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
