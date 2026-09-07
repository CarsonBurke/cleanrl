"""Reward-normalized PPO critic proxy, calibrated after the v2 MuJoCo regression.

Run via mlq. CPU, seed1, fixed work. Unlike v2: support is independent of target
mean/variation, targets have positive offsets and small state contrasts, gamma
is .99, every trajectory state trains the critic, and detached lambda labels
are reused for ten full-batch epochs. Confirmation uses the actual sphere trunk.

This is a critic-only, fixed-policy MDP experiment, not a MuJoCo score predictor.
The actor-gradient probes include PPO clipping and no advantage normalization;
no joint actor/critic optimizer or policy-induced visitation changes are modeled.
No objective is ranked by CE or target entropy. Report paired sigma contrasts
and raw metrics rather than promoting a single pooled universal winner.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleanrl.shared.hl_gauss import HLGaussConfig
from cleanrl.shared.host_actor import make_situ_sphere_trunk
from scripts.hlgauss.diagnostics import analyze_support


@dataclass(frozen=True)
class Candidate:
    loss: str = "gaussian"
    bins: int = 31
    bound: float = 50.0
    sigma: float = 0.75
    geometry: Literal["centers", "symexp_centers"] = "symexp_centers"
    transform: Literal["linear", "symlog"] = "linear"

    @property
    def key(self):
        if self.loss != "gaussian":
            return self.loss
        return f"{self.geometry}/{self.transform}/k{self.bins}/b{self.bound:g}/s{self.sigma:g}"

    @property
    def outputs(self):
        return self.bins if self.loss == "gaussian" else 1

    def config(self):
        return HLGaussConfig(
            v_min=-self.bound,
            v_max=self.bound,
            num_bins=self.bins,
            sigma_ratio=self.sigma,
            bin_type=self.geometry,
            transform=self.transform,
            decode="scalar",
        )


@dataclass(frozen=True)
class Case:
    name: str
    mean: float
    value_std: float
    reward_noise: float
    lr: float
    gamma: float = 0.99
    shift: bool = False
    frequency: int = 1


# Values are rounded calibrations, not estimates of hidden baseline value means.
# Late noise is an ablation: logged critic residuals do not identify aleatoric noise.
CASES = (
    Case("early", 1.0, 1.0, 0.03, 0.0096),
    Case("late", 3.8337, 0.3858, 0.006, 0.004224),
    Case("late_noisy", 3.8337, 0.3858, 0.06, 0.004224),
    Case("offset_shift_holdout", 6.0, 0.2, 0.006, 0.004224, shift=True, frequency=2),
    Case("broad_noisy_control", 0.0, 2.0, 0.2, 0.003, gamma=0.97),
)


class EnsembleCritic:
    """Independent leaf parameter banks; vmap only changes execution layout."""

    def __init__(self, count, outputs, architecture, gaussian):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            if architecture == "sphere":
                trunk = make_situ_sphere_trunk(6, 64)
                head = nn.Linear(64, outputs)
                if not gaussian:
                    nn.init.orthogonal_(head.weight, 1.0)
            else:
                trunk = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
                head = nn.Linear(32, outputs)
            nn.init.zeros_(head.bias)
            if gaussian:
                nn.init.zeros_(head.weight)
            prototype = nn.Sequential(trunk, head)
        self.params, self.buffers = stack_module_state([prototype] * count)
        self.template = prototype.to("meta")
        self.forward = vmap(lambda p, b, x: functional_call(self.template, (p, b), (x,)), in_dims=(0, 0, None))

    def __call__(self, x):
        return self.forward(self.params, self.buffers, x)

    def parameters(self):
        return self.params.values()


def make_mdp(case, phase=0):
    n = 96
    angle = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    x = torch.stack(
        [angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos(), (3 * angle).sin(), (3 * angle).cos()], -1
    )
    p = torch.zeros(2, n, n)
    for action, direction in enumerate((-1, 1)):
        # Low transition noise is separate from the explicit reward-noise
        # ablation. A 5% global teleport overwhelms the late critic residual
        # scale and recreates the old proxy's noisy-regression regime.
        for distance, probability in ((1, 0.95), (3, 0.049)):
            p[action, torch.arange(n), (torch.arange(n) + direction * distance) % n] += probability
    p += 0.001 / n
    shape = torch.sin(case.frequency * angle + phase * 0.9) + 0.3 * torch.cos(3 * angle)
    shape = (shape - shape.mean()) / shape.std(unbiased=False)
    value = case.mean + phase * 0.5 + case.value_std * shape
    # Small nonzero action contrasts; exact zero-mean bonus under the .5 policy.
    bonus = case.value_std * 0.04 * (angle.cos() + 0.3 * torch.sin(3 * angle))
    advantage = torch.stack((-bonus, bonus))
    reward = value[None] - case.gamma * (p @ value) + advantage
    return x, p, reward, value, advantage


def sample_rollout(case, phase, horizon, envs, generator):
    x, p, reward, truth, _ = make_mdp(case, phase)
    states = [torch.randint(len(x), (envs,), generator=generator)]
    actions, rewards = [], []
    for _ in range(horizon):
        action = torch.randint(2, (envs,), generator=generator)
        current = states[-1]
        noise = torch.randn(envs, generator=generator) * case.reward_noise
        actions.append(action)
        rewards.append(reward[action, current] + noise)
        states.append(torch.multinomial(p[action, current], 1, generator=generator).squeeze(-1))
    return x, torch.stack(states), torch.stack(actions), torch.stack(rewards), truth


def lambda_advantages(rewards, values, gamma, lam=0.95):
    """values: (models,H+1,envs), rewards: (H,envs); all states are consumed."""
    result = torch.empty_like(values[:, :-1])
    carry = torch.zeros_like(values[:, 0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t][None] + gamma * values[:, t + 1] - values[:, t]
        carry = delta + gamma * lam * carry
        result[:, t] = carry
    return result


def ppo_gradients(x, actions, advantages, normalize=False):
    """Exact sample-surrogate gradients for a linear Bernoulli actor at 3 probes.

    Old policy is .5. Fixed probes are the same for oracle and every candidate;
    they exercise on-policy and already-clipped PPO branches. Gradients are of
    the loss, not of a separately normalized or reweighted advantage estimator.
    """
    if normalize:
        advantages = (advantages - advantages.mean(-1, keepdim=True)) / (
            advantages.std(-1, keepdim=True, unbiased=False) + 1e-8
        )
    direction = torch.tensor([0.8, -0.5, 0.3, 0.2, -0.2, 0.1], dtype=x.dtype)
    gradients = []
    for amplitude in (0.0, 0.6, -0.6):
        prob = (x @ (amplitude * direction)).sigmoid()
        ratio = 2 * torch.where(actions.bool(), prob, 1 - prob)
        active = torch.where(advantages >= 0, ratio <= 1.2, ratio >= 0.8)
        factor = -advantages * active * ratio * (actions - prob)
        gradients.append(factor @ x / len(x))
    return torch.stack(gradients, dim=1)


def metrics(predictions, truth, x, states, actions, rewards, case):
    oracle_adv = lambda_advantages(rewards, truth[states].unsqueeze(0), case.gamma).flatten(1)
    predicted_adv = lambda_advantages(rewards, predictions[:, states], case.gamma).flatten(1)
    error = predicted_adv - oracle_adv
    obs = x[states[:-1].flatten()]
    acts = actions.flatten().float()
    reference = ppo_gradients(obs, acts, oracle_adv)
    actual = ppo_gradients(obs, acts, predicted_adv)
    ref_norm = reference.square().sum((1, 2)).clamp_min(1e-12)
    grad_error = (actual - reference).square().sum((1, 2)) / ref_norm
    ref_normalized = ppo_gradients(obs, acts, oracle_adv, normalize=True)
    actual_normalized = ppo_gradients(obs, acts, predicted_adv, normalize=True)
    normalized_error = (actual_normalized - ref_normalized).square().sum((1, 2)) / ref_normalized.square().sum(
        (1, 2)
    ).clamp_min(1e-12)
    adv_std = oracle_adv.std(unbiased=False).clamp_min(1e-8)
    informative = oracle_adv.abs() > 0.1 * adv_std
    signs = ((predicted_adv > 0) != (oracle_adv > 0)) & informative
    mse = (predictions - truth).square().mean(-1)
    return dict(
        value_nmse=mse / case.value_std**2,
        oracle_advantage_std=adv_std.expand(predictions.shape[0]),
        oracle_return_std=(oracle_adv + truth[states[:-1]].flatten()).std(unbiased=False).expand(predictions.shape[0]),
        value_bias=(predictions - truth).mean(-1),
        advantage_relative_mse=error.square().mean(-1) / adv_std.square(),
        advantage_bias=error.mean(-1),
        advantage_bias_in_std=error.mean(-1) / adv_std,
        advantage_sign_error=signs.float().sum(-1) / informative.sum(),
        ppo_gradient_relative_mse=grad_error,
        ppo_gradient_relative_mse_advnorm=normalized_error,
    )


def decode(logits, heads):
    return torch.stack([h.to_scalar(logits[i]) for i, h in enumerate(heads)]) if heads else logits.squeeze(-1)


def update_critics(model, optimizer, observations, labels, old_values, clipped_mse):
    logits = model(observations)
    if clipped_mse is not None:
        values = logits.squeeze(-1)
        squared = (values - labels).square()
        clipped = old_values + (values - old_values).clamp(-0.2, 0.2)
        losses = 0.5 * torch.where(clipped_mse, torch.maximum(squared, (clipped - labels).square()), squared).mean(-1)
    else:
        losses = -(labels * logits.log_softmax(-1)).sum(-1).mean(-1)
    optimizer.zero_grad(set_to_none=True)
    (0.5 * losses.sum()).backward()
    norm = torch.stack([p.grad.flatten(1).square().sum(-1) for p in model.parameters()]).sum(0).sqrt()
    multiplier = (0.5 / (norm + 1e-6)).clamp(max=1)
    for p in model.parameters():
        p.grad.mul_(multiplier.reshape(-1, *([1] * (p.ndim - 1))))
    optimizer.step()
    return norm.detach()


def train_group(
    candidates,
    case,
    architecture,
    rollouts,
    horizon=128,
    envs=4,
    bootstrap="own",
    *,
    sampler=sample_rollout,
    initialize=None,
):
    heads = [c.config().build("cpu") for c in candidates] if candidates[0].loss == "gaussian" else []
    model = EnsembleCritic(len(candidates), candidates[0].outputs, architecture, bool(heads))
    if initialize is not None:
        initialize(model, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=case.lr, eps=1e-5)
    generator = torch.Generator().manual_seed(1)
    # Evaluation trajectories use a separate fixed stream. They never train.
    evaluation = [sampler(case, phase, horizon, envs * 4, torch.Generator().manual_seed(9001)) for phase in range(2)]
    curves = [[] for _ in candidates]
    overflow = torch.zeros(len(candidates))
    critic_norms = torch.zeros(len(candidates))
    clipped_mse = None if heads else torch.tensor([c.loss == "mse_clipped" for c in candidates])[:, None]
    clipped_updates = torch.zeros(len(candidates))
    start = time.perf_counter()
    for rollout in range(rollouts):
        phase = int(case.shift and rollout >= 20)
        x, states, _, rewards, truth = sampler(case, phase, horizon, envs, generator)
        with torch.no_grad():
            old = decode(model(x), heads)
            bootstrap_values = truth[None].expand_as(old) if bootstrap == "oracle" else old
            adv = lambda_advantages(rewards, bootstrap_values[:, states], case.gamma)
            target = (adv + bootstrap_values[:, states[:-1]]).flatten(1)
            old_flat = old[:, states[:-1]].flatten(1)
            labels = torch.stack([head.project(target[i]) for i, head in enumerate(heads)]) if heads else target
            if heads:
                for i, c in enumerate(candidates):
                    overflow[i] += (target[i].abs() > c.bound).float().mean()
        observations = x[states[:-1].flatten()]
        for _ in range(10):
            norm = update_critics(model, optimizer, observations, labels, old_flat, clipped_mse)
            critic_norms += norm
            clipped_updates += (norm > 0.5).float()
        if (rollout + 1) % 5 == 0 or rollout == rollouts - 1:
            with torch.no_grad():
                ex, es, ea, er, ev = evaluation[phase]
                measured = metrics(decode(model(ex), heads), ev, ex, es, ea, er, case)
                for i, c in enumerate(candidates):
                    curves[i].append(
                        dict(rollout=rollout + 1, phase=phase, **{k: float(v[i]) for k, v in measured.items()})
                    )
    elapsed = time.perf_counter() - start
    rows = []
    for i, c in enumerate(candidates):
        curve = curves[i]
        tail = curve[-max(1, len(curve) // 4) :]
        rows.append(
            dict(
                candidate=c.key,
                case=case.name,
                architecture=architecture,
                bootstrap=bootstrap,
                lr=case.lr,
                **{k: v for k, v in curve[-1].items() if k not in ("rollout", "phase")},
                **{
                    f"tail_{k}": sum(p[k] for p in tail) / len(tail) for k in curve[-1] if k not in ("rollout", "phase")
                },
                tail_checkpoint_count=len(tail),
                mean_checkpoint_value_nmse=sum(p["value_nmse"] for p in curve) / len(curve),
                mean_checkpoint_ppo_gradient_mse=sum(p["ppo_gradient_relative_mse"] for p in curve) / len(curve),
                mean_critic_gradient_norm=float(critic_norms[i] / (10 * rollouts)),
                critic_clip_fraction=float(clipped_updates[i] / (10 * rollouts)),
                support_overflow_fraction=float(overflow[i] / rollouts),
                amortized_seconds=elapsed / len(candidates),
                curve=curve,
            )
        )
    return rows


def grouped(candidates, width=10):
    for _, group in itertools.groupby(
        sorted(candidates, key=lambda c: (c.loss == "gaussian", c.outputs)),
        key=lambda c: (c.loss == "gaussian", c.outputs),
    ):
        group = list(group)
        for i in range(0, len(group), width):
            yield group[i : i + width]


def paired_contrasts(rows, candidates):
    configs = {c.key: c for c in candidates}
    indexed = {(r["candidate"], r["case"], r["architecture"], r["bootstrap"]): r for r in rows}
    result = []
    for row in rows:
        c = configs[row["candidate"]]
        if c.loss != "gaussian" or c.sigma not in (0.5, 0.75):
            continue
        reference = Candidate(bins=c.bins, bound=c.bound, sigma=2.0, geometry=c.geometry, transform=c.transform)
        old = indexed.get((reference.key, row["case"], row["architecture"], row["bootstrap"]))
        if old is None:
            continue
        result.append(
            dict(
                candidate=c.key,
                reference=reference.key,
                case=row["case"],
                architecture=row["architecture"],
                bootstrap=row["bootstrap"],
                value_nmse_ratio=row["tail_value_nmse"] / max(old["tail_value_nmse"], 1e-12),
                ppo_gradient_mse_ratio=row["tail_ppo_gradient_relative_mse"]
                / max(old["tail_ppo_gradient_relative_mse"], 1e-12),
                advantage_mse_ratio=row["tail_advantage_relative_mse"] / max(old["tail_advantage_relative_mse"], 1e-12),
            )
        )
    return result


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
    geometries: tuple[tuple[Literal["centers", "symexp_centers"], Literal["linear", "symlog"]], ...] = (
        ("centers", "linear"),
        ("symexp_centers", "linear"),
        ("centers", "symlog"),
    )
    candidates = [
        Candidate(bins=k, bound=b, sigma=s, geometry=g, transform=t)
        for k, b, s, (g, t) in itertools.product(
            (31, 101),
            (10.0, 50.0),
            (0.25, 0.5, 0.75, 1.0, 2.0),
            geometries,
        )
    ]
    candidates += [Candidate(loss="mse"), Candidate(loss="mse_clipped")]
    # Confirmation roster is declared before observing scores, not selected for
    # matching the user's historical outcome. Keep all paired sigmas/geometries.
    confirmation = [c for c in candidates if c.loss != "gaussian" or (c.bins == 31 and c.sigma in (0.5, 0.75, 2.0))]
    args.output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.output))
    result: dict[str, Any] = dict(
        protocol_version=3,
        protocol_revision=2,
        initial_calibration_job=5097,
        shift_rollout=20,
        paired_metric_window="mean of final quarter of evenly spaced checkpoints",
        seed=1,
        device="cpu",
        threads=1,
        rollouts=args.rollouts,
        horizon=128,
        num_envs=4,
        epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        candidates={c.key: asdict(c) for c in candidates},
        cases=[asdict(c) for c in CASES],
        confirmation_roster=[c.key for c in confirmation],
        diagnostics=[],
        training=[],
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("scripts/hlgauss/diagnostics.py"),
                Path("cleanrl/shared/hl_gauss.py"),
                Path("cleanrl/shared/host_actor.py"),
            )
        },
        limitations=[
            "Fixed-policy critic-only MDP, not MuJoCo/closed-loop PPO returns",
            "No joint actor-critic gradient clipping coupling or policy visitation shifts",
            "Observed target std is total variation, not a measured conditional noise estimate",
            "128-step rolls approximate long PPO lambda returns; not2048x16 production batch",
            "One seed; no between-run significance or universal-default claim",
            "tanh screening and sphere confirmation use separately stated case sets",
        ],
    )

    def save():
        temporary = args.output / "results.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(args.output / "results.json")
        writer.flush()

    start = time.perf_counter()
    for case in CASES:
        for c in candidates:
            if c.loss == "gaussian":
                result["diagnostics"].append(
                    dict(
                        candidate=c.key,
                        case=case.name,
                        **analyze_support(c.config(), case.mean, case.value_std, gamma=case.gamma),
                    )
                )
    save()
    print(f'DIAGNOSTICS {len(result["diagnostics"])} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    for architecture, roster, cases, bootstrap in (
        ("tanh", candidates, (CASES[0], CASES[1], CASES[4]), "own"),
        ("sphere", confirmation, (CASES[1], CASES[2], CASES[3]), "own"),
        ("sphere", confirmation, (CASES[1],), "oracle"),
    ):
        for case in cases:
            for group in grouped(roster):
                rows = train_group(group, case, architecture, args.rollouts, bootstrap=bootstrap)
                result["training"].extend(rows)
                for row in rows:
                    writer.add_scalar(
                        f'{architecture}/{bootstrap}/{case.name}/{row["candidate"]}/ppo_gradient_relative_mse',
                        row["tail_ppo_gradient_relative_mse"],
                        args.rollouts * 10,
                    )
            save()
            print(
                f"FIT {architecture} {bootstrap} {case.name} configs={len(roster)} elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
    result["paired_sigma_contrasts"] = paired_contrasts(result["training"], candidates)
    result["elapsed_seconds"] = time.perf_counter() - start
    save()
    writer.close()
    for row in result["training"]:
        if row["architecture"] == "sphere" and row["case"] == "late":
            print(
                f'FINAL {row["bootstrap"]} {row["candidate"]} value={row["tail_value_nmse"]:.6g} advantage={row["tail_advantage_relative_mse"]:.6g} ppo_grad={row["tail_ppo_gradient_relative_mse"]:.6g} adv_bias={row["tail_advantage_bias"]:.6g}',
                flush=True,
            )
    print(f'RESULT {args.output/"results.json"} elapsed={result["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == "__main__":
    main()
