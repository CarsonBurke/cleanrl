"""Test whether larger uniform categorical heads are the right response to support.

CPU/seed1 critic proxy, queued through mlq; no MuJoCo or actor training. Reuses
160-rollout sphere critics, own-bootstrap lambda targets, ten fixed-label epochs,
and clipped, unnormalized sampled PPO-gradient probes. Predeclared comparisons,
not a post-hoc default-selection sweep. Bounds-only, offset, broad, and rare-value
stressors are separate. Barycentric Gaussian labels preserve the raw mean;
uniform barycentric labels control for changing projection as well as geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleanrl.shared.hl_gauss import HLGaussConfig
from scripts.hlgauss.ppo_proxy_v3 import (
    Candidate,
    Case,
    EnsembleCritic,
    grouped,
    lambda_advantages,
    sample_rollout,
    train_group,
    update_critics,
)
from scripts.hlgauss.support_projection import MeanPreservingSupport


@dataclass(frozen=True)
class SupportCase(Case):
    tail_height: float = 0.0


def sample_case(case, phase, horizon, envs, generator):
    """Potential-shaped tail: V'=V+f(s), r'=r+f(s)-gamma*f(s').

    For oracle V, every sampled TD residual and lambda advantage is unchanged.
    The critic must nevertheless represent the rare high values. This is a
    localized value tail, not a claim about the noise law of MuJoCo returns.
    """
    x, states, actions, rewards, truth = sample_rollout(case, phase, horizon, envs, generator)
    if case.tail_height:
        potential = case.tail_height * (500 * (x[:, 1] - 1)).exp()
        rewards = rewards + potential[states[:-1]] - case.gamma * potential[states[1:]]
        truth = truth + potential
    return x, states, actions, rewards, truth


def calibrate(case) -> tuple[dict[str, Any], torch.Tensor]:
    _, states, _, rewards, truth = sample_case(case, 0, 128, 32, torch.Generator().manual_seed(7001))
    targets = lambda_advantages(rewards, truth[states][None], case.gamma) + truth[states[:-1]][None]
    return (
        dict(
            target_std=float(targets.std(unbiased=False)),
            target_median=float(targets.median()),
            target_quantiles=[float(v) for v in targets.quantile(torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0]))],
            true_value_std=float(truth.std(unbiased=False)),
        ),
        targets.flatten(),
    )


@dataclass(frozen=True)
class Spec:
    projector: str
    bins: int
    bound: float
    center: float
    scale: float
    geometry: str = "uniform"
    smoothing: str = "fixed"
    loss: str = "gaussian"

    @property
    def outputs(self):
        return self.bins

    @property
    def key(self):
        return f"{self.projector}/{self.geometry}/{self.smoothing}/k{self.bins}/b{self.bound:g}"

    def config(self):
        return self

    def build(self, device="cpu"):
        if self.projector == "histogram":
            return HLGaussConfig(v_min=-self.bound, v_max=self.bound, num_bins=self.bins).build(device)
        return MeanPreservingSupport(
            v_min=-self.bound,
            v_max=self.bound,
            num_bins=self.bins,
            center=self.center,
            scale=self.scale,
            geometry=self.geometry,
            smoothing=self.smoothing,
        ).to(device)


POLICIES = (
    "capped_uniform",
    "uncapped_uniform",
    "uniform_bary101",
    "asinh_bary101",
    "asinh_bary31",
    "asinh_local101",
    "asinh_twohot101",
)


def resolve(bound, calibration):
    scale, center = calibration["target_std"], calibration["target_median"]
    current = HLGaussConfig.for_target_scale(v_min=-bound, v_max=bound, target_std=scale)
    uncapped_bins = max(3, 2 * math.ceil(bound / (2 * scale)) + 1)
    result = {
        "capped_uniform": Spec("histogram", current.num_bins, bound, center, scale),
        "uncapped_uniform": Spec("histogram", uncapped_bins, bound, center, scale),
    }
    for policy, geometry, bins, smoothing in (
        ("uniform_bary101", "uniform", 101, "fixed"),
        ("asinh_bary101", "asinh", 101, "fixed"),
        ("asinh_bary31", "asinh", 31, "fixed"),
        ("asinh_local101", "asinh", 101, "local"),
        ("asinh_twohot101", "asinh", 101, "twohot"),
    ):
        result[policy] = Spec("barycentric", bins, bound, center, scale, geometry, smoothing)
    return result


def initial_logits(support):
    """Maximum-entropy prior constrained to raw E[V]=0; no target-informed mean.

    Symmetric uniform supports already satisfy this with zero logits. Asymmetric
    adaptive supports require a nonuniform prior to avoid an initialization win.
    """
    coordinates = support.double() / support.abs().max()
    low, high = -64.0, 64.0
    for _ in range(64):
        tilt = (low + high) / 2
        mean = float((torch.softmax(tilt * coordinates, -1) * coordinates).sum())
        if mean < 0:
            low = tilt
        else:
            high = tilt
    logits = ((low + high) / 2 * coordinates).to(support.dtype)
    error = float((logits.softmax(-1) * support).sum().abs())
    if error > 2e-7 * float(support.abs().max()):
        raise ArithmeticError(f"initial value is not zero: {error}")
    return logits


def initialize(model, heads):
    if heads:
        with torch.no_grad():
            model.params["1.bias"].copy_(torch.stack([initial_logits(h.support) for h in heads]))


def numerical_diagnostics(spec, targets):
    head = spec.build()
    interior = targets.clamp(-0.95 * spec.bound, 0.95 * spec.bound)
    probes = torch.cat((interior, torch.linspace(-spec.bound, spec.bound, 513)))
    labels = head.project(probes)
    error = head.probs_to_scalar(labels) - probes
    occupied_error = error[: len(interior)]
    return dict(
        occupied_projection_bias=float(occupied_error.mean()),
        occupied_projection_rmse=float(occupied_error.square().mean().sqrt()),
        entire_support_max_abs_projection_error=float(error.abs().max()),
        max_probability_sum_error=float((labels.sum(-1) - 1).abs().max()),
        minimum_probability=float(labels.min()),
        initial_value=float(head.to_scalar(initial_logits(head.support))),
        output_head_parameters=65 * spec.bins,
        fp32_batch_logits_and_labels_bytes=2 * 512 * spec.bins * 4,
    )


def benchmark_cost(spec, targets):
    """Controlled one-candidate CPU timings, independent of fit-group batching.

    Same 512 states, full sphere trunk and Adam, 10 updates per label projection.
    Report CPU latency, not a GPU/runtime-optimality claim. Warm optimizer state
    before timing; repeated windows reduce timer noise without fitting outcomes.
    """
    head = spec.build()
    model = EnsembleCritic(1, spec.bins, "sphere", True)
    initialize(model, [head])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004224, eps=1e-5)
    generator = torch.Generator().manual_seed(8101)
    observations = torch.randn(512, 6, generator=generator)
    target = targets[:512]
    labels = head.project(target).unsqueeze(0)
    for _ in range(5):
        update_critics(model, optimizer, observations, labels, None, None)
    projection, updates = [], []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):
            head.project(target)
        projection.append((time.perf_counter() - start) / 10)
        start = time.perf_counter()
        for _ in range(10):
            update_critics(model, optimizer, observations, labels, None, None)
        updates.append((time.perf_counter() - start) / 10)
    label_seconds, update_seconds = statistics.median(projection), statistics.median(updates)
    return dict(
        label_seconds=label_seconds,
        update_seconds=update_seconds,
        projection_plus_ten_updates_seconds=label_seconds + 10 * update_seconds,
        repetitions=5,
        inner_repetitions=10,
        device="cpu",
        batch_size=512,
    )


def rank(cells):
    result = []
    for policy in POLICIES:
        relative = [
            c["policies"][policy]["tail_ppo_gradient_relative_mse"]
            / max(c["scalar"]["tail_ppo_gradient_relative_mse"], 1e-12)
            for c in cells
        ]
        result.append(
            dict(
                policy=policy,
                cells=len(cells),
                geometric_mean_scalar_relative_error=math.exp(
                    statistics.mean(math.log(max(v, 1e-12)) for v in relative)
                ),
                worst_scalar_relative_error=max(relative),
            )
        )
    return sorted(result, key=lambda r: r["geometric_mean_scalar_relative_error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.output))
    cases = (
        (SupportCase("bounds_only", 3.8337, 0.3858, 0.006, 0.004224), (10.0, 50.0, 200.0, 1000.0)),
        (SupportCase("offset_only", 30.0, 0.3858, 0.006, 0.004224), (200.0,)),
        (SupportCase("broader_values", 3.8337, 4.0, 0.06, 0.004224), (200.0,)),
        (SupportCase("rare_value_tail", 3.8337, 0.3858, 0.006, 0.004224, tail_height=20.0), (200.0,)),
    )
    report: dict[str, Any] = dict(
        protocol="support-geometry-v1",
        seed=1,
        device="cpu",
        rollouts=160,
        policies=list(POLICIES),
        primary_new_hypothesis="asinh_bary101",
        cells=[],
        costs={},
        predeclared_cases=[dict(case=asdict(case), bounds=bounds) for case, bounds in cases],
        calibration="4096 oracle-bootstrap returns, independent seed7001; median and total std, not noise estimates",
        initialization="categorical maximum-entropy distributions with raw expectation zero",
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("scripts/hlgauss/support_projection.py"),
                Path("scripts/hlgauss/ppo_proxy_v3.py"),
                Path("cleanrl/shared/hl_gauss.py"),
                Path("cleanrl/shared/host_actor.py"),
            )
        },
        limitations=[
            "Single-seed synthetic fixed-policy critic evidence, not MuJoCo scores or actor/critic feedback",
            "Oracle-bootstrap calibration is not a deployable cold-start estimator",
            "Asinh allocates bins using both calibration median and std; all methods get identical raw bounds",
            "Gaussian barycentric labels change the projection, controlled by uniform_bary101",
            "Fixed-bandwidth barycentric sigma is 1.5*target_std; local sigma is .75*enclosing-cell width",
            "Rare tails are localized potential-shaped values, not heavy-tailed reward noise",
            "CPU timing is controlled, but is not GPU throughput; fit amortized timings are not ranked",
            "No held-out selection stage; configurations and case cells are declared before any fitting",
        ],
    )
    start = time.perf_counter()

    def save():
        temp = args.output / "results.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(args.output / "results.json")
        writer.flush()

    save()
    for case, bounds in cases:
        calibration, targets = calibrate(case)
        mappings = {bound: resolve(bound, calibration) for bound in bounds}
        unique = {s.key: s for mapping in mappings.values() for s in mapping.values()}
        roster = [Candidate(loss="mse_clipped"), *unique.values()]
        results = {}
        for group in grouped(roster):
            for row in train_group(group, case, "sphere", 160, sampler=sample_case, initialize=initialize):
                # Tail cases have a larger true-value variance than the base MDP.
                for metric in ("value_nmse", "tail_value_nmse", "mean_checkpoint_value_nmse"):
                    row[metric] *= case.value_std**2 / calibration["true_value_std"] ** 2
                for point in row["curve"]:
                    point["value_nmse"] *= case.value_std**2 / calibration["true_value_std"] ** 2
                results[row["candidate"]] = row
                writer.add_scalar(
                    f"progress/{case.name}/{row['candidate']}/gradient_error",
                    row["tail_ppo_gradient_relative_mse"],
                    1600,
                )
            writer.flush()
            print(
                f"FIT {case.name} models={len(group)} bins={group[0].outputs} elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
        for bound, mapping in mappings.items():
            diagnostics = {s.key: numerical_diagnostics(s, targets) for s in mapping.values()}
            report["cells"].append(
                dict(
                    case=asdict(case),
                    bound=bound,
                    calibration=calibration,
                    scalar=results["mse_clipped"],
                    policies={policy: results[s.key] for policy, s in mapping.items()},
                    configs={policy: asdict(s) for policy, s in mapping.items()},
                    diagnostics={policy: diagnostics[s.key] for policy, s in mapping.items()},
                )
            )
            if case.name == "bounds_only":
                costs = {s.key: benchmark_cost(s, targets) for s in {s.key: s for s in mapping.values()}.values()}
                report["costs"][str(bound)] = {policy: costs[s.key] for policy, s in mapping.items()}
        save()
        print(f"CASE {case.name} cells={len(report['cells'])} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    report["ranking"] = rank(report["cells"])
    report["elapsed_seconds"] = time.perf_counter() - start
    save()
    writer.close()
    for row in report["ranking"]:
        print("RANK " + json.dumps(row), flush=True)
    print(f"RESULT {args.output/'results.json'} elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
