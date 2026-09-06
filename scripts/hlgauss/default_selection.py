"""Select a raw-uniform HL-Gauss default jointly over bandwidth and head resolution.

CPU/seed1; run through mlq. Reuses the corrected sphere/PPO proxy unchanged.
Fixed K/sigma policies compete with resolution chosen from an independent
calibration batch's return standard deviation, under the same 255-bin head
budget. Absolute sigma always equals sigma_ratio * raw bin width. The budget
is an explicit capacity constraint, not a target clamp. No moving supports.

Selection uses geometric-mean clipped-PPO gradient error relative to scalar MSE
across all declared case/support cells. Lock best fixed and scale-aware policies
before two new teacher/scale cases; retain matched sigma controls in confirmation.
This selects a proxy default, not a universally optimal MuJoCo configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.hlgauss.ppo_proxy_v3 import (
    CASES,
    Candidate,
    Case,
    grouped,
    lambda_advantages,
    sample_rollout,
    train_group,
)


@dataclass(frozen=True)
class Policy:
    bins: int = 101
    sigma: float = 0.75
    width_in_std: float | None = None
    max_bins: int = 255
    auto_sigma: bool = False

    @property
    def key(self):
        resolution = f"k{self.bins}" if self.width_in_std is None else f"stdwidth{self.width_in_std:g}_budget{self.max_bins}"
        return f"{resolution}/bias_limited_sigma" if self.auto_sigma else f"{resolution}/s{self.sigma:g}"

    def candidate(self, bound, target_std):
        bins = self.bins
        if self.width_in_std is not None:
            bins = min(self.max_bins, max(3, 2 * math.ceil(bound / (target_std * self.width_in_std)) + 1))
        sigma = self.sigma
        if self.auto_sigma:
            width = 2 * bound / (bins - 1)
            # Infinite uniform-grid mean-quantization error is bounded by
            # width/pi * q/(1-q), q=exp(-2*pi²*ratio²). Choose the sharpest
            # tested kernel below 0.5% of target std; not a tail-bias guarantee.
            for sigma in (0.5, 0.75, 1.0, 2.0):
                q = math.exp(-2 * math.pi**2 * sigma**2)
                if width / math.pi * q / (1 - q) <= 0.005 * target_std:
                    break
        return Candidate(bins=bins, bound=bound, sigma=sigma, geometry="centers", transform="linear")


def calibration_std(case):
    _, states, _, rewards, truth = sample_rollout(case, 0, 128, 32, torch.Generator().manual_seed(7001))
    targets = lambda_advantages(rewards, truth[states].unsqueeze(0), case.gamma) + truth[states[:-1]][None]
    return float(targets.std(unbiased=False))


def rank(rows, policies) -> list[dict[str, Any]]:
    scores = {p.key: [] for p in policies}
    timings = {p.key: [] for p in policies}
    for cell in rows:
        reference = cell["scalar"]["tail_ppo_gradient_relative_mse"]
        for key, row in cell["policies"].items():
            scores[key].append(row["tail_ppo_gradient_relative_mse"] / max(reference, 1e-12))
            timings[key].append(row["amortized_seconds"])
    return sorted(
        (
            dict(
                policy=key,
                geometric_mean_relative_gradient_mse=math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values)),
                worst_relative_gradient_mse=max(values),
                cells=len(values),
                mean_seconds=sum(timings[key]) / len(values),
            )
            for key, values in scores.items()
        ),
        key=lambda row: row["geometric_mean_relative_gradient_mse"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=160)
    args = parser.parse_args()
    if args.rollouts < 40 or args.rollouts % 10:
        parser.error("rollouts must be >=40 and a multiple of10")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)
    policies = [Policy(bins=k, sigma=s) for k in (31, 101, 255) for s in (0.5, 0.75, 1.0, 2.0)]
    policies += [Policy(width_in_std=w, sigma=s) for w in (0.5, 1.0, 2.0) for s in (0.5, 0.75, 1.0, 2.0)]
    policies += [Policy(bins=k, auto_sigma=True) for k in (31, 101, 255)]
    args.output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.output))
    report: dict[str, Any] = dict(
        protocol="joint-default-selection-v1",
        seed=1,
        device="cpu",
        rollouts=args.rollouts,
        policies={p.key: asdict(p) for p in policies},
        screening=[],
        confirmation=[],
        calibration="4096 independent oracle-bootstrap targets, seed7001; observable return std, not noise estimation",
        capacity_budget=255,
        selection_metric="geometric mean of final-quarter PPO-gradient MSE / scalar-control error across case/support cells",
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("scripts/hlgauss/ppo_proxy_v3.py"),
                Path("cleanrl/shared/hl_gauss.py"),
                Path("cleanrl/shared/host_actor.py"),
            )
        },
        limitations=[
            "One-seed synthetic fixed-policy evidence; not a MuJoCo score optimum",
            "Calibration uses oracle-bootstrap returns, unavailable during cold-start PPO; a representative target scale must be supplied explicitly",
            "255-bin ceiling is a declared head-capacity budget; oversized supports may remain resolution-limited",
            "Amortized CPU timings share overhead across variable group sizes; not controlled per-policy or GPU throughput",
            "Bias-limited sigma bounds infinite-grid quantization only, not finite-support truncation or learning error",
        ],
    )
    start = time.perf_counter()

    def save():
        temp = args.output / "results.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(args.output / "results.json")
        writer.flush()

    def evaluate(stage, case, bounds, roster):
        scale = calibration_std(case)
        mapping = {(bound, p.key): p.candidate(bound, scale) for bound in bounds for p in roster}
        unique = {c.key: c for c in mapping.values()}
        unique["mse_clipped"] = Candidate(loss="mse_clipped")
        results = {}
        for group in grouped(list(unique.values())):
            for row in train_group(group, case, "sphere", args.rollouts):
                results[row["candidate"]] = row
        for bound in bounds:
            cell: dict[str, Any] = dict(
                case=asdict(case),
                bound=bound,
                calibrated_target_std=scale,
                scalar=results["mse_clipped"],
                policies={p.key: results[mapping[(bound, p.key)].key] for p in roster},
                resolved_configs={p.key: asdict(mapping[(bound, p.key)]) for p in roster},
            )
            report[stage].append(cell)
            for p in roster:
                writer.add_scalar(
                    f"{stage}/{case.name}/bound{bound:g}/{p.key}/ppo_gradient_error",
                    cell["policies"][p.key]["tail_ppo_gradient_relative_mse"],
                    args.rollouts * 10,
                )
        save()
        print(
            f"FIT {stage} {case.name} unique={len(unique)} target_std={scale:.6g} elapsed={time.perf_counter()-start:.1f}s",
            flush=True,
        )

    for case in CASES:
        evaluate("screening", case, (10.0, 50.0), policies)
    report["screening_ranking"] = rank(report["screening"], policies)
    by_key = {p.key: p for p in policies}
    fixed = next(
        by_key[r["policy"]]
        for r in report["screening_ranking"]
        if by_key[r["policy"]].width_in_std is None and not by_key[r["policy"]].auto_sigma
    )
    adaptive = next(by_key[r["policy"]] for r in report["screening_ranking"] if by_key[r["policy"]].width_in_std is not None)
    sigma_adaptive = next(by_key[r["policy"]] for r in report["screening_ranking"] if by_key[r["policy"]].auto_sigma)
    finalists = list(
        {
            p.key: p
            for base in (fixed, adaptive, sigma_adaptive)
            for p in [Policy(bins=base.bins, width_in_std=base.width_in_std, sigma=s) for s in (0.5, 0.75, 1.0, 2.0)]
        }.values()
    )
    finalists.append(sigma_adaptive)
    report["policies"].update({p.key: asdict(p) for p in finalists})
    report["locked_finalists"] = [p.key for p in finalists]
    save()
    print("LOCKED " + json.dumps(report["locked_finalists"]), flush=True)
    for case in (
        Case("fine_scale_confirmation", 2.0, 0.15, 0.005, 0.004224, frequency=3),
        Case("broad_scale_confirmation", 8.0, 2.0, 0.05, 0.004224, frequency=2, shift=True),
    ):
        evaluate("confirmation", case, (20.0, 100.0), finalists)
    report["confirmation_ranking"] = rank(report["confirmation"], finalists)
    report["elapsed_seconds"] = time.perf_counter() - start
    save()
    writer.close()
    for stage in ("screening", "confirmation"):
        for row in report[stage + "_ranking"]:
            print(f"RANK {stage} " + json.dumps(row), flush=True)
    print(f'RESULT {args.output/"results.json"} elapsed={report["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == "__main__":
    main()
