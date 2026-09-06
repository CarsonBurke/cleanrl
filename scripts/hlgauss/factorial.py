"""Seed-1 CPU critic proxy: factorial screening, untouched confirmation, TD usefulness.

Run through mlq (not a MuJoCo score claim). Bounds are RAW units. Screening uses
held-out conditional means, never incomparable categorical CE. Confirmation uses
new teacher/data streams; finalists are locked before it is generated. One seed
means descriptive paired evidence, not statistical significance across RL runs.

Historical protocol: its sigma-2 joint winner regressed in normalized-reward
PPO. TD utility here cancels constant critic bias; matched nonuniform sigmas
were not all retained through confirmation. Use scripts/hlgauss/ppo_proxy_v3.py
for the reward-normalized, raw-GAE/clipped-PPO comparison, not these rankings
as a universal smoothing recommendation. Keep this protocol for reproducibility.

The fixed-update budget measures sample/update efficiency; timings expose the
extra cost of larger heads. Identical initial trunks, minibatches and noises are
used for every candidate. Independent batched MLPs are only an execution device:
losses are summed across models and gradient norms clipped per model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleanrl.shared.hl_gauss import (
    Dreamer3BucketHLGaussSupport,
    HLGaussCDFSupport,
    HLGaussConfig,
    symexp,
    symlog,
)


@dataclass(frozen=True)
class Candidate:
    kind: str = "gaussian"
    bins: int = 101
    sigma: float = 0.75
    transform: Literal["linear", "symlog"] = "linear"
    geometry: Literal["centers", "edges", "symexp_centers"] = "centers"
    decode: Literal["scalar", "transformed"] = "scalar"
    headroom: float = 3.0

    @property
    def outputs(self):
        return 1 if self.kind == "mse" else self.bins - int(self.kind == "cdf")

    @property
    def key(self):
        return f"{self.kind}/{self.transform}/{self.geometry}/{self.decode}/k{self.bins}/s{self.sigma}/h{self.headroom}"

    def build(self, scale):
        bound = scale * self.headroom
        ratio = self.sigma
        if self.kind == "fixed_sigma":
            # Fix absolute coordinate sigma while varying K and headroom.
            coord_scale = math.log1p(scale) if self.transform == "symlog" else scale
            coord_bound = math.log1p(bound) if self.transform == "symlog" else bound
            ratio = 0.05 * coord_scale / (2 * coord_bound / (self.bins - 1))
        return HLGaussConfig(
            num_bins=self.bins,
            v_min=-bound,
            v_max=bound,
            sigma_ratio=ratio,
            transform=self.transform,
            bin_type=self.geometry,
            decode=self.decode,
        ).build("cpu")


REFERENCE_CLASSES = {
    "raw_grid_symlog": (
        "cleanrl/iterthink/v24_d4hlgauss/other/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawbins_v1.py",
        "RawSpacedSymlogHLGaussSupport",
    ),
    "symexp_grid_raw": (
        "cleanrl/iterthink/wm_k6/ppo_continuous_action_iterthink_v152_2_d3linbins_k6.py",
        "D3LinearHLGaussSupport",
    ),
}


@lru_cache(maxsize=None)
def reference_class(kind):
    """Execute only the trusted frozen class, not its PPO entrypoint/imports."""
    path, name = REFERENCE_CLASSES[kind]
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace: dict[str, Any] = {"torch": torch, "np": np, "symlog": symlog, "symexp": symexp}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class Labels:
    def __init__(self, candidate, scale):
        self.candidate, self.scale = candidate, scale
        self.support: Any = candidate.build(scale) if candidate.kind != "mse" else None
        self.legacy = None
        if candidate.kind == "moment_matched":
            bound = math.log1p(scale * candidate.headroom)
            self.legacy = Dreamer3BucketHLGaussSupport(candidate.bins, -bound, bound, candidate.sigma, "cpu")
            self.support = self.legacy
        elif candidate.kind == "raw_grid_symlog":
            bound = scale * candidate.headroom
            self.support = reference_class(candidate.kind)(candidate.bins, -bound, bound, candidate.sigma, "cpu")
        elif candidate.kind == "cdf":
            bound = scale * candidate.headroom
            if candidate.transform == "symlog":
                bound = math.log1p(bound)
            self.support = HLGaussCDFSupport(
                candidate.bins, -bound, bound, candidate.sigma, "cpu", use_symlog=candidate.transform == "symlog"
            )

    def project(self, y):
        c = self.candidate
        if c.kind == "mse":
            return (symlog(y) if c.transform == "symlog" else y / self.scale).unsqueeze(-1)
        if self.legacy is not None:
            return self.legacy.project_moment_matched(y)
        assert self.support is not None
        if c.kind == "cdf":
            return self.support.cdf_labels(y)
        if c.kind == "twohot":
            # Raw-space barycentric interpolation preserves E[value] exactly.
            centers = self.support.support
            y = y.clamp(centers[0], centers[-1])
            upper = torch.searchsorted(centers, y.contiguous()).clamp(1, c.bins - 1)
            lower = upper - 1
            weight = (y - centers[lower]) / (centers[upper] - centers[lower])
            result = torch.zeros(*y.shape, c.bins)
            result.scatter_(-1, lower.unsqueeze(-1), (1 - weight).unsqueeze(-1))
            return result.scatter_add_(-1, upper.unsqueeze(-1), weight.unsqueeze(-1))
        return self.support.project(y)

    def decode(self, logits):
        if self.candidate.kind == "mse":
            z = logits.squeeze(-1)
            return symexp(z) if self.candidate.transform == "symlog" else z * self.scale
        assert self.support is not None
        return self.support.to_scalar(logits)


class Critics(nn.Module):
    def __init__(self, count, inputs, outputs, seed=1):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for index, (left, right) in enumerate(zip((inputs, 32, 32), (32, 32, outputs))):
            # Matching trunk draws regardless of output dimension or ensemble size.
            weight = torch.randn(left, right, generator=generator) / math.sqrt(left)
            if index == 2:
                weight *= 0.01
            self.weights.append(nn.Parameter(weight.unsqueeze(0).repeat(count, 1, 1)))
            self.biases.append(nn.Parameter(torch.zeros(count, 1, right)))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.weights[0].shape[0], -1, -1)
        for index, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, w) + b
            if index != 2:
                x = x.tanh()
        return x


def update(model, optimizer, x, labels, mse, cdf=False):
    logits = model(x)
    if mse:
        losses = (logits - labels).square().mean((1, 2))
    elif cdf:
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean((1, 2))
    else:
        losses = -(labels * logits.log_softmax(-1)).sum(-1).mean(-1)
    optimizer.zero_grad(set_to_none=True)
    losses.sum().backward()
    # Never let one candidate's large gradients clip another candidate's model.
    norm = torch.stack([p.grad.flatten(1).square().sum(-1) for p in model.parameters()]).sum(0).sqrt()
    multiplier = (0.5 / (norm + 1e-6)).clamp(max=1)
    for p in model.parameters():
        p.grad.mul_(multiplier.reshape(-1, *([1] * (p.ndim - 1))))
    optimizer.step()


def teacher(x, family, phase, rotation):
    a = x @ rotation
    if family == "smooth":
        return 0.65 * torch.tanh(a[:, 0]) + 0.25 * torch.sin(2 * a[:, 1])
    if family == "skewed":
        return 0.1 + 0.8 * torch.sigmoid(2 * a[:, 0]) + 0.1 * torch.sin(a[:, 1])
    if family == "lottery":
        probability = 0.05 + 0.4 * torch.sigmoid(a[:, 0])
        payoff = 0.05 + 0.9 * torch.sigmoid(2 * a[:, 1])
        return probability * payoff
    return (0.65 * torch.tanh(a[:, phase]) + 0.25 * torch.sin(a[:, 2])) * (1 if phase == 0 else -1)


def regression_data(scale, family, split):
    g = torch.Generator().manual_seed(1 + split * 10000 + ("smooth", "skewed", "moving", "lottery").index(family) * 100)
    rotation = torch.linalg.qr(torch.randn(6, 6, generator=g)).Q
    x = torch.randn(2048, 6, generator=g)
    test_x = torch.randn(1024, 6, generator=g)
    targets, test_means = [], []
    for phase in range(2):
        mu = teacher(x, family, phase, rotation) * scale
        noise_scale = (0.1 + 0.25 * torch.sigmoid(x[:, 0])) * scale
        if family == "smooth":
            noise = torch.randn(len(x), generator=g)
        else:
            # E[noise|x]=0, but the transformed conditional mean is not symlog(mu).
            rare = torch.rand(len(x), generator=g) < 0.1
            noise = torch.where(rare, 3.0, -1 / 3) + 0.1 * torch.randn(len(x), generator=g)
        if family == "lottery":
            a = x @ rotation
            probability = 0.05 + 0.4 * torch.sigmoid(a[:, 0])
            payoff = scale * (0.05 + 0.9 * torch.sigmoid(2 * a[:, 1]))
            targets.append((torch.rand(len(x), generator=g) < probability) * payoff)
        else:
            targets.append(mu + noise_scale * noise)
        test_means.append(teacher(test_x, family, phase, rotation) * scale)
    indices = torch.randint(len(x), (1000, 128), generator=g)
    return x, targets, test_x, test_means, indices


def regression(candidates, scale, family, split, steps, lr):
    x, targets, test_x, test_means, indices = regression_data(scale, family, split)
    encoders = [Labels(c, scale) for c in candidates]
    model = Critics(len(candidates), x.shape[-1], candidates[0].outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    projection_start = time.perf_counter()
    labels = [torch.stack([e.project(y) for e in encoders]) for y in targets]
    projection_seconds = (time.perf_counter() - projection_start) / len(candidates)
    curves = [[] for _ in candidates]
    start = time.perf_counter()
    with torch.no_grad():
        oracle = []
        for c, e in zip(candidates, encoders):
            if e.support is None:
                oracle.append(0.0)
            else:
                probs = e.project(test_means[-1])
                prediction = e.decode(torch.logit(probs) if c.kind == "cdf" else probs.log())
                oracle.append(float(((prediction - test_means[-1]) / scale).square().mean()))
    for step in range(steps + 1):
        phase = int(family == "moving" and step >= steps // 2)
        if step % 20 == 0 or step == steps:
            with torch.no_grad():
                logits = model(test_x)
                # The switch is a discontinuity: record both sides at the SAME
                # update count so trapezoids do not smear it into the old phase.
                phases = (0, 1) if family == "moving" and step == steps // 2 else (phase,)
                for current_phase in phases:
                    for i, e in enumerate(encoders):
                        error = (e.decode(logits[i]) - test_means[current_phase]) / scale
                        curves[i].append(
                            {
                                "step": step,
                                "phase": current_phase,
                                "nmse": float(error.square().mean()),
                                "bias": float(error.mean()),
                            }
                        )
        if step == steps:
            break
        idx = indices[step]
        update(model, optimizer, x[idx], labels[phase][:, idx], candidates[0].kind == "mse", candidates[0].kind == "cdf")
    seconds = (time.perf_counter() - start) / len(candidates)
    rows = []
    for i, (c, e) in enumerate(zip(candidates, encoders)):
        clipped = (
            0 if e.support is None else float((targets[int(family == "moving")].abs() > scale * c.headroom).float().mean())
        )
        curve = curves[i]
        # Trapezoidal integral; the initial error belongs in sample efficiency.
        area = sum((a["nmse"] + b["nmse"]) * (b["step"] - a["step"]) / 2 for a, b in zip(curve, curve[1:])) / steps
        rows.append(
            dict(
                candidate=c.key,
                task=f"{family}_{scale:g}",
                split=split,
                lr=lr,
                final_nmse=curve[-1]["nmse"],
                auc_nmse=area,
                bias=curve[-1]["bias"],
                noiseless_projection_nmse=oracle[i],
                clipping_fraction=clipped,
                projection_seconds=projection_seconds,
                amortized_seconds=seconds,
                parameters=sum(p[0].numel() for p in model.parameters()),
                curve=curve,
            )
        )
    return rows


def make_mrp(scale):
    n = 96
    angle = torch.arange(n) * (2 * math.pi / n)
    x = torch.stack([angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos(), (3 * angle).sin(), (3 * angle).cos()], -1)
    transitions = torch.zeros(2, n, n)
    for action, direction in enumerate((-1, 1)):
        for offset, probability in ((1, 0.7), (3, 0.25)):
            transitions[action, torch.arange(n), (torch.arange(n) + direction * offset) % n] += probability
    transitions += 0.05 / n
    value = scale * (0.55 * angle.sin() + 0.25 * (2 * angle).cos())
    gamma = 0.97
    bonus = scale * 0.06 * (angle.cos() + 0.3 * (3 * angle).sin())
    advantage = torch.stack((-bonus, bonus))
    reward = value[None] - gamma * (transitions @ value) + advantage
    return x, transitions, reward, value, advantage, gamma


def td_metrics(prediction, transitions, reward, truth, advantage, gamma, scale):
    estimated_advantage = reward + gamma * (transitions @ prediction) - prediction[None]
    difference = estimated_advantage - advantage
    # Actual one-step actor utility, followed by exact evaluation of the new policy.
    # Both actor updates use the same oracle advantage scale to isolate critic error.
    adv_scale = advantage.square().mean().sqrt()
    policy = (0.5 * (estimated_advantage[1] - estimated_advantage[0]) / adv_scale).sigmoid()
    oracle_policy = (0.5 * (advantage[1] - advantage[0]) / adv_scale).sigmoid()

    def improvement(p):
        p_transition = (1 - p[:, None]) * transitions[0] + p[:, None] * transitions[1]
        p_reward = (1 - p) * reward[0] + p * reward[1]
        value = torch.linalg.solve(torch.eye(len(p), dtype=torch.float64) - gamma * p_transition.double(), p_reward.double())
        return float((value - truth).mean())

    delta = estimated_advantage[1] - estimated_advantage[0]
    true_delta = advantage[1] - advantage[0]
    return dict(
        value_nmse=float(((prediction - truth) / scale).square().mean()),
        advantage_relative_mse=float(difference.square().mean() / advantage.square().mean()),
        action_sign_accuracy=float(((delta > 0) == (true_delta > 0)).float().mean()),
        policy_improvement=improvement(policy),
        oracle_policy_improvement=improvement(oracle_policy),
        policy_utility_ratio=improvement(policy) / improvement(oracle_policy),
    )


def temporal_difference(candidates, scale, steps, lr):
    x, transition, reward, truth, advantage, gamma = make_mrp(scale)
    encoders = [Labels(c, scale) for c in candidates]
    model = Critics(len(candidates), 6, candidates[0].outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    g = torch.Generator().manual_seed(900001)
    curves = [[] for _ in candidates]
    clip_counts = torch.zeros(len(candidates))
    # 8-step lambda returns, frozen for four minibatch passes, include actual
    # sampled transition/reward noise rather than oracle values as labels.
    for rollout in range(steps // 4):
        states = [torch.randint(len(x), (128,), generator=g)]
        rewards = []
        for _ in range(8):
            action = torch.randint(2, (128,), generator=g)
            current = states[-1]
            rewards.append(reward[action, current] + 0.03 * scale * torch.randn(128, generator=g))
            states.append(torch.multinomial(transition[action, current], 1, generator=g).squeeze(-1))
        with torch.no_grad():
            logits = model(x)
            values = torch.stack([e.decode(logits[i]) for i, e in enumerate(encoders)])
            target = values[:, states[-1]]
            for t in reversed(range(8)):
                target = rewards[t][None] + gamma * (0.05 * values[:, states[t + 1]] + 0.95 * target)
            labels = torch.stack([e.project(target[i]) for i, e in enumerate(encoders)])
            for i, c in enumerate(candidates):
                clip_counts[i] += (target[i].abs() > scale * c.headroom).float().mean() if c.kind != "mse" else 0
        for _ in range(4):
            update(model, optimizer, x[states[0]], labels, candidates[0].kind == "mse", candidates[0].kind == "cdf")
        if (rollout + 1) % 10 == 0 or rollout == steps // 4 - 1:
            with torch.no_grad():
                logits = model(x)
                for i, e in enumerate(encoders):
                    metrics = td_metrics(e.decode(logits[i]), transition, reward, truth, advantage, gamma, scale)
                    curves[i].append(dict(step=(rollout + 1) * 4, **metrics))
    return [
        dict(
            candidate=c.key,
            scale=scale,
            lr=lr,
            **curves[i][-1],
            clipping_fraction=float(clip_counts[i] / (steps // 4)),
            curve=curves[i],
        )
        for i, c in enumerate(candidates)
    ]


def groups(candidates, max_group=12):
    key = lambda c: (c.kind == "mse", c.kind == "cdf", c.bins)
    for _, group in itertools.groupby(sorted(candidates, key=key), key=key):
        group = list(group)
        for i in range(0, len(group), max_group):
            yield group[i : i + max_group]


def ranking(rows) -> list[dict[str, Any]]:
    collected = {}
    for row in rows:
        # Equal task weight, units normalized by the generating value scale.
        collected.setdefault(row["candidate"], []).append(0.5 * (row["final_nmse"] + row["auc_nmse"]))
    return sorted(
        (dict(candidate=key, score=sum(values) / len(values), worst=max(values)) for key, values in collected.items()),
        key=lambda r: r["score"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 80 or args.steps > 1000 or args.steps % 40:
        parser.error("steps must be a multiple of 40 in [80,1000]")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.output))
    candidates = [
        Candidate(bins=k, sigma=s, transform=t, geometry=g, decode=d, headroom=h)
        for k, s, t, g, h in itertools.product(
            (31, 101, 255), (0.5, 0.75, 1.5, 2.0), ("linear", "symlog"), ("centers", "edges"), (1.1, 3.0, 10.0)
        )
        for d in (("scalar", "transformed") if t == "symlog" else ("scalar",))
    ]
    controls = [Candidate(kind="mse", bins=1, transform=t) for t in ("linear", "symlog")]
    controls += [
        Candidate(kind="twohot", bins=k, transform=t, headroom=h)
        for k, t, h in itertools.product((31, 101, 255), ("linear", "symlog"), (1.1, 3.0, 10.0))
    ]
    controls += [
        Candidate(kind="moment_matched", bins=k, transform="symlog", sigma=s, headroom=h)
        for k, s, h in itertools.product((31, 101, 255), (0.75, 2.0), (1.1, 3.0, 10.0))
    ]
    controls += [
        Candidate(
            kind=kind,
            bins=k,
            transform="linear" if kind == "symexp_grid_raw" else "symlog",
            geometry="symexp_centers" if kind == "symexp_grid_raw" else "edges",
            sigma=s,
            headroom=h,
        )
        for kind, k, s, h in itertools.product(REFERENCE_CLASSES, (31, 101, 255), (0.75, 2.0), (1.1, 3.0))
    ]
    controls += [
        Candidate(kind="cdf", bins=k, transform=t, decode="transformed", sigma=s, headroom=h)
        for k, t, s, h in itertools.product((31, 101, 255), ("linear", "symlog"), (0.75, 2.0), (1.1, 3.0))
    ]
    controls += [
        Candidate(kind="fixed_sigma", bins=k, transform=t, headroom=h)
        for k, t, h in itertools.product((31, 101, 255), ("linear", "symlog"), (1.1, 3.0, 10.0))
    ]
    candidates += controls
    report: dict[str, Any] = dict(
        protocol_version=2,
        seed=1,
        device="cpu",
        steps=args.steps,
        threads=args.threads,
        torch=torch.__version__,
        python=platform.python_version(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("cleanrl/shared/hl_gauss.py"),
                *(Path(spec[0]) for spec in REFERENCE_CLASSES.values()),
            )
        },
        limitations=[
            "Synthetic critic proxy, not MuJoCo performance",
            "One seed; no between-run significance claim",
            "Fixed updates, not equal wall time; head sizes differ",
            "Symmetric supports; raw scale is known to the proxy",
            "Finalist selection uses screening only; confirmation is descriptive, not another search",
        ],
        candidates={c.key: asdict(c) for c in candidates},
        screening=[],
        confirmation=[],
        td=[],
    )

    def save():
        temporary = args.output / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(args.output / "results.json")
        writer.flush()

    started = time.perf_counter()
    cases = list(itertools.product((5.0, 50.0, 500.0), ("smooth", "skewed", "moving", "lottery")))
    for scale, family in cases:
        for batch in groups(candidates):
            rows = regression(batch, scale, family, 0, args.steps, 0.003)
            report["screening"].extend(rows)
            for row in rows:
                writer.add_scalar(f"screen/{row['task']}/{row['candidate']}/nmse", row["final_nmse"], args.steps)
        save()
        print(
            f"SCREEN {family} scale={scale} candidates={len(candidates)} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    ranked = ranking(report["screening"])
    report["screening_ranking"] = ranked
    # Lock global top six plus best of every objective/transform, not handpicked
    # winners after seeing confirmation. Always retain both scalar MSE controls.
    chosen = {r["candidate"] for r in ranked[:6]}
    for kind, transform in itertools.product(
        ("gaussian", "twohot", "moment_matched", "mse", "cdf", "fixed_sigma", *REFERENCE_CLASSES), ("linear", "symlog")
    ):
        match = next(
            (
                r
                for r in ranked
                if report["candidates"][r["candidate"]]["kind"] == kind
                and report["candidates"][r["candidate"]]["transform"] == transform
            ),
            None,
        )
        if match:
            chosen.add(match["candidate"])
    # Keep the best of each Gaussian geometry/decoder pairing even if it loses
    # screening: confirmation must still expose the estimand distinction.
    for transform, geometry, decode in itertools.product(
        ("linear", "symlog"), ("centers", "edges"), ("scalar", "transformed")
    ):
        match = next(
            (
                r
                for r in ranked
                if all(
                    report["candidates"][r["candidate"]][k] == v
                    for k, v in (("kind", "gaussian"), ("transform", transform), ("geometry", geometry), ("decode", decode))
                )
            ),
            None,
        )
        if match:
            chosen.add(match["candidate"])
    finalists = [c for c in candidates if c.key in chosen]
    report["locked_finalists"] = [c.key for c in finalists]
    save()
    print("LOCKED " + json.dumps(report["locked_finalists"]), flush=True)
    for lr in (0.001, 0.003, 0.01):
        for scale, family in cases:
            for batch in groups(finalists):
                report["confirmation"].extend(regression(batch, scale, family, 1, args.steps, lr))
        for scale in (5.0, 50.0, 500.0):
            for batch in groups(finalists):
                report["td"].extend(temporal_difference(batch, scale, args.steps, lr))
        save()
        print(f"CONFIRM lr={lr} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    report["confirmation_ranking"] = ranking(report["confirmation"])
    report["elapsed_seconds"] = time.perf_counter() - started
    for i, row in enumerate(report["confirmation_ranking"]):
        writer.add_scalar(f"confirmation/{row['candidate']}/score", row["score"], args.steps)
        print(f"RANK {i+1} {row['score']:.6f} {row['candidate']}", flush=True)
    for row in report["td"]:
        writer.add_scalar(f"td/{row['scale']}/{row['lr']}/{row['candidate']}/utility", row["policy_utility_ratio"], args.steps)
    save()
    writer.close()
    print(f"RESULT {args.output / 'results.json'} elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
