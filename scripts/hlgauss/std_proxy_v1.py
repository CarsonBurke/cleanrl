"""Estimator-error proxy for the target-standardized HL-Gauss head.

This is NOT a return predictor, and it is deliberately not built on
``scripts/hlgauss/ppo_proxy_v3.py``. That harness ranks candidates by the
squared error of a surrogate actor gradient on a fixed-policy, stationary,
exactly representable MDP; over its own 10 screening cells no HL-Gauss policy
beat the scalar control, its sigma ranking inverts between own-bootstrap and
oracle-bootstrap in the same cell, and its scalar control's value clip is a raw
0.2 that lands anywhere from 0.09 to 1.3 target standard deviations depending
on the cell. The one configuration it ever selected prospectively regressed in
MuJoCo.

What this proxy measures instead is the single quantity a value head is
responsible for: recovering the conditional mean of noisy, drifting regression
targets. It fixes the confounds that made the previous sigma rankings
uninterpretable:

* per-candidate learning-rate grid, so sigma is not ranked by the effective
  step size it induces through ``dCE/dvalue = dMSE/dvalue / Var_p(z)``;
* the scalar control's value clip is expressed in target standard deviations
  (0.51, production's 0.2 against a 0.39 target spread), not raw units;
* the target location and scale drift across iterations over the range the
  reward-normalized HalfCheetah critic actually sees (mean 0 -> 4, spread
  0.25 -> 0.6), so a static support is not silently rewarded;
* scored against the true conditional mean on held-out states, so fitting the
  target noise is penalized rather than rewarded;
* a falsification anchor: the shipped absolute-support geometry (+-50, 101
  bins, sigma 0.75 raw = 1.9 target std) is in the roster. It is known to lose
  in MuJoCo. A proxy that ranks it well has failed, and its sigma ranking
  should not be used.

Limits, stated up front: no actor, no bootstrapping, no policy-induced
distribution shift, one seed per cell. It can rank label geometry. It cannot
establish a return.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from cleanrl.shared.hl_gauss import HLGaussConfig
from cleanrl.shared.hl_gauss_std import StandardizedHistogram
from cleanrl.shared.norm_residual import make_norm_residual_trunk

OBS_DIM = 17
WIDTH = 64
# Measured from the reward-normalized sphere runs: the cross-state return
# spread and its mean both grow by roughly 2.4x and 4 absolute over training.
DRIFT_MEAN = (0.0, 4.0)
DRIFT_SCALE = (0.25, 0.6)
CLIP_IN_TARGET_STD = 0.51


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str  # scalar | std_hlgauss | std_mse | absolute_hlgauss
    bins: int = 101
    sigma_over_std: float = 0.15
    half_span: float = 5.0
    clip: bool = True


@dataclass(frozen=True)
class Cell:
    name: str
    noise_ratio: float
    noise: str  # normal | t3
    drift: bool = True


@dataclass
class Fit:
    nmse: float = math.inf
    slope: float = 0.0
    learning_rate: float = 0.0
    history: list = field(default_factory=list)


def teacher(device, seed):
    """A fixed smooth function of the state, standardized to unit spread."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randn((OBS_DIM, 128), generator=generator) / math.sqrt(OBS_DIM)
    second = torch.randn((128, 1), generator=generator) / math.sqrt(128)
    first, second = first.to(device), second.to(device)

    def value(states):
        hidden = torch.tanh(states @ first)
        raw = (hidden**3 @ second).squeeze(-1)
        return raw

    probe = value(torch.randn((16384, OBS_DIM), device=device))
    centre, spread = probe.mean(), probe.std()
    return lambda states: (value(states) - centre) / spread


def schedule(iteration, iterations, drift):
    if not drift:
        return DRIFT_MEAN[1], DRIFT_SCALE[1]
    fraction = iteration / max(iterations - 1, 1)
    return (
        DRIFT_MEAN[0] + fraction * (DRIFT_MEAN[1] - DRIFT_MEAN[0]),
        DRIFT_SCALE[0] + fraction * (DRIFT_SCALE[1] - DRIFT_SCALE[0]),
    )


def draw_noise(shape, kind, device, generator):
    if kind == "normal":
        return torch.randn(shape, device=device, generator=generator)
    # Student-t(3), standardized: heavy-tailed targets are the regime the
    # classification loss is claimed to be robust in.
    normal = torch.randn(shape, device=device, generator=generator)
    chi = torch.randn((*shape, 3), device=device, generator=generator).square().sum(-1)
    return (normal / (chi / 3).sqrt()) / math.sqrt(3.0)


class Critic(nn.Module):
    def __init__(self, outputs, head_gain, device):
        super().__init__()
        self.trunk = make_norm_residual_trunk(
            OBS_DIM, WIDTH, placement="pre", norm_kind="rms", activation="stiglu"
        )
        self.head = nn.Linear(WIDTH, outputs)
        nn.init.orthogonal_(self.head.weight, head_gain)
        nn.init.zeros_(self.head.bias)
        self.to(device)

    def forward(self, states):
        return self.head(self.trunk(states))


SCALAR_KINDS = ("scalar", "scalar_std", "scalar_std_bounded")
REGRESSION_KINDS = SCALAR_KINDS + ("std_mse",)


def build(candidate, device):
    if candidate.kind == "scalar":
        return Critic(1, 1.0, device), None
    if candidate.kind == "absolute_hlgauss":
        support = HLGaussConfig(
            v_min=-50.0, v_max=50.0, num_bins=candidate.bins, sigma_ratio=0.75, bin_type="centers"
        ).build(device)
        return Critic(candidate.bins, 0.01, device), support
    histogram = StandardizedHistogram(
        candidate.bins,
        half_span=candidate.half_span,
        sigma_bins=candidate.sigma_over_std / (2.0 * candidate.half_span / (candidate.bins - 1)),
        device=device,
    )
    if candidate.kind in SCALAR_KINDS:
        # Same location/scale bookkeeping, scalar head: isolates the free
        # absorption of global target drift from the categorical loss itself.
        return Critic(1, 1.0, device), histogram
    return Critic(candidate.bins, 0.01, device), histogram


def decode(candidate, support, readout):
    if candidate.kind == "scalar":
        return readout.squeeze(-1)
    if candidate.kind in SCALAR_KINDS:
        raw = readout.squeeze(-1)
        if candidate.kind == "scalar_std_bounded":
            raw = candidate.half_span * torch.tanh(raw / candidate.half_span)
        return support.mean + support.scale * raw
    if candidate.kind == "absolute_hlgauss":
        return support.to_scalar(readout)
    return support.decode(readout)


def predict(candidate, model, support, states):
    return decode(candidate, support, model(states))


def value_loss(candidate, model, support, states, targets, old_values, clip):
    readout = model(states)
    value = decode(candidate, support, readout)
    if candidate.kind in REGRESSION_KINDS:
        squared = (value - targets).square()
        if not candidate.clip:
            return 0.5 * squared.mean()
        clipped = old_values + (value - old_values).clamp(-clip, clip)
        return 0.5 * torch.maximum(squared, (clipped - targets).square()).mean()
    cross_entropy = -(support.project(targets) * readout.log_softmax(-1)).sum(-1)
    if candidate.clip:
        with torch.no_grad():
            clipped = old_values + (value - old_values).clamp(-clip, clip)
            frozen = ((clipped - targets).square() > (value - targets).square()).float()
        cross_entropy = cross_entropy * (1.0 - frozen)
    return cross_entropy.mean()


def run_fit(candidate, cell, learning_rate, args, device):
    torch.manual_seed(args.seed)
    model, support = build(candidate, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    truth = teacher(device, args.seed)
    stream = torch.Generator(device=device).manual_seed(args.seed + 17)
    holdout = torch.randn((args.eval_size, OBS_DIM), device=device, generator=stream)
    history = []
    for iteration in range(args.iterations):
        centre, spread = schedule(iteration, args.iterations, cell.drift)
        states = torch.randn((args.batch, OBS_DIM), device=device, generator=stream)
        with torch.no_grad():
            exact = centre + spread * truth(states)
            noise = draw_noise((args.batch,), cell.noise, device, stream)
            targets = exact + cell.noise_ratio * spread * noise
            if support is not None and isinstance(support, StandardizedHistogram):
                support.observe(targets)
            old_values = predict(candidate, model, support, states)
        clip = CLIP_IN_TARGET_STD * spread
        for _ in range(args.epochs):
            loss = value_loss(candidate, model, support, states, targets, old_values, clip)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        with torch.no_grad():
            reference = centre + spread * truth(holdout)
            predicted = predict(candidate, model, support, holdout)
            error = ((predicted - reference).square().mean() / spread**2).item()
            centered = reference - reference.mean()
            slope = ((predicted - predicted.mean()) * centered).sum() / centered.square().sum()
            history.append((error, slope.item()))
    tail = history[-max(1, args.iterations // 4) :]
    return Fit(
        nmse=sum(row[0] for row in tail) / len(tail),
        slope=sum(row[1] for row in tail) / len(tail),
        learning_rate=learning_rate,
        history=[round(row[0], 6) for row in history],
    )


def roster():
    candidates = [
        Candidate("scalar_mse_clipped", "scalar"),
        Candidate("scalar_mse_unclipped", "scalar", clip=False),
        Candidate("std_mse_softmax_k101", "std_mse", bins=101, sigma_over_std=0.15),
        Candidate("absolute_hlgauss_k101_pm50", "absolute_hlgauss", bins=101),
        # Decisive controls: a scalar head with the *same* per-rollout location
        # and scale bookkeeping. If these match the categorical arms, the gain
        # is target standardization, and classification contributes nothing.
        Candidate("scalar_mse_popart", "scalar_std"),
        Candidate("scalar_mse_popart_bounded", "scalar_std_bounded", half_span=5.0),
    ]
    for bins in (51, 101, 201):
        for sigma in (0.0375, 0.075, 0.15, 0.3, 0.5):
            candidates.append(
                Candidate(f"std_hlgauss_k{bins}_s{sigma}", "std_hlgauss", bins=bins, sigma_over_std=sigma)
            )
    return candidates


def cells():
    return [
        Cell("light_normal", 0.25, "normal"),
        Cell("heavy_normal", 1.0, "normal"),
        Cell("light_t3", 0.25, "t3"),
        Cell("heavy_t3", 1.0, "t3"),
        Cell("heavy_normal_static", 1.0, "normal", drift=False),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--eval-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="benchmarks/hlgauss/std_proxy_v1.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    learning_rates = (1e-3, 3e-3, 9.6e-3)
    started = time.perf_counter()
    results = {}
    for cell in cells():
        per_cell = {}
        for candidate in roster():
            best = Fit()
            for learning_rate in learning_rates:
                fit = run_fit(candidate, cell, learning_rate, args, device)
                if fit.nmse < best.nmse:
                    best = fit
            per_cell[candidate.name] = {
                "nmse": best.nmse,
                "slope": best.slope,
                "learning_rate": best.learning_rate,
                "history": best.history,
            }
            print(f"{cell.name:22s} {candidate.name:28s} nmse={best.nmse:.5f} lr={best.learning_rate}")
        results[cell.name] = per_cell
    ranking = []
    for name in (candidate.name for candidate in roster()):
        relative = [
            results[cell.name][name]["nmse"] / max(results[cell.name]["scalar_mse_clipped"]["nmse"], 1e-12)
            for cell in cells()
        ]
        ranking.append(
            {
                "candidate": name,
                "geometric_mean_relative_nmse": math.exp(
                    sum(math.log(max(value, 1e-12)) for value in relative) / len(relative)
                ),
                "worst_relative_nmse": max(relative),
                "per_cell_relative_nmse": dict(zip((cell.name for cell in cells()), relative)),
            }
        )
    ranking.sort(key=lambda row: row["geometric_mean_relative_nmse"])
    anchor = next(row for row in ranking if row["candidate"] == "absolute_hlgauss_k101_pm50")
    record = {
        "status": "completed",
        "measures": "held-out conditional-mean NMSE under noisy, drifting targets; not returns",
        "seconds": time.perf_counter() - started,
        "config": vars(args),
        "roster": [candidate.name for candidate in roster()],
        "learning_rates": list(learning_rates),
        "ranking": ranking,
        "falsification_anchor": {
            "candidate": anchor["candidate"],
            "geometric_mean_relative_nmse": anchor["geometric_mean_relative_nmse"],
            "rank": ranking.index(anchor) + 1,
            "rule": (
                "The shipped absolute-support geometry lost in MuJoCo. If it ranks in the top "
                "third here, this proxy has no discriminating power and its sigma ranking must "
                "not be used to choose a run."
            ),
        },
        "cells": results,
    }
    with open(args.output, "w") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps({row["candidate"]: round(row["geometric_mean_relative_nmse"], 4) for row in ranking}, indent=1))


if __name__ == "__main__":
    main()
