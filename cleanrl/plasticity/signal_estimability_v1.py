"""Evaluator-only information diagnostic; no outputs are supplied to learners.

Replay the exact seed-1, 100-observation RNG chunks of the batch-one benchmark.
Fit an optimistic support-shape-informed scalar amplitude to noisy observations.
This is a diagnostic reference, not a deployable optimizer or universal bound.
"""
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity.noisy_stream_diagnostic import Args, Stream


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    directory = Path("runs") / f"signal_estimability_v1__1__{time.time():.6f}"
    directory.mkdir(parents=True)
    results = {}
    for name, count, signal_inputs, switch_at, switch_back in (
        ("sparse", 20000, 1, 0.0, 0.0),
        ("dense", 20000, 4096, 0.0, 0.0),
        ("switch", 100000, 1, 0.5, 0.75),
    ):
        args = Args(steps=count, seeds=1, seed=1, signal_inputs=signal_inputs,
                    switch_at=switch_at, switch_back=switch_back, switch_to=2)
        stream = Stream(args, torch.device("cuda"), False)
        chunks = {}
        for start in range(0, count, 100):
            _, target, clean, _, _ = stream.draw(100, start + 1)
            phase = "initial" if not switch_at or start < count * switch_at else (
                "moved" if start < count * switch_back else "returned")
            chunks.setdefault(phase, []).append(torch.stack((clean[:, 0], target[:, 0]), dim=1))
        phases = {}
        for phase, pieces in chunks.items():
            values = torch.cat(pieces).double()
            clean, target = values.unbind(1)
            information = clean.square().sum()
            amplitude = (clean * target).sum() / information
            p = args.feature_prob
            clean_variance = signal_inputs * p * (1 - p)
            clean_power = clean_variance + (signal_inputs * p) ** 2
            risk = (amplitude - 1).square() * clean_power
            row = {
                "observations": len(values), "active_observations": int((clean != 0).sum()),
                "support_shape_oracle_amplitude": float(amplitude),
                "support_shape_oracle_clean_risk": float(risk),
                "support_shape_oracle_clean_R2": float(1 - risk / clean_variance),
                "gaussian_amplitude_standard_error": float(math.sqrt(5) / information.sqrt()),
                "gaussian_unbiased_scalar_risk_reference": float(5 * clean_power / information),
                "clean_variance": clean_variance, "clean_power": clean_power,
                "caveat": "Uses true support shape, with no discovery cost. Gaussian reference omits rare spikes, ignores change discovery, and is not a universal bound for biased estimators.",
            }
            if signal_inputs == 1:
                active_targets = target[clean != 0]
                row["active_target_median"] = float(active_targets.median())
            phases[phase] = row
        results[name] = phases
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with SummaryWriter(str(directory)) as writer:
        for task, phases in results.items():
            for phase, values in phases.items():
                writer.add_scalar(f"{task}/{phase}/oracle_clean_R2", values["support_shape_oracle_clean_R2"], values["observations"])
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved {directory / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
