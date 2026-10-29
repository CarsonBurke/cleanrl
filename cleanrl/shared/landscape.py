"""Return and surrogate landscapes of a frozen policy along parameter-space directions.

A policy family is described by an ``Adapter``: the network that maps normalized observations to
distribution outputs, the snapshot's frozen observation normalization, how outputs and a base noise draw
become physical actions at a noise amplitude, named parameter subsets, and optionally a Q critic. Every
grid point is evaluated with common random numbers: episode e resets with ``seed + e`` and draws the same
base noise at every step for every point, so differences between points are paired and far less noisy
than the returns themselves.

The noise amplitude ``scale`` multiplies the policy's own noise around its center action (0 is the
deterministic policy, 1 the trained one): a direct measurement of the true cost of exploration noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, vmap

Params = dict[str, torch.Tensor]


@dataclass
class Adapter:
    module: nn.Module
    """maps normalized observations [n, obs] to distribution outputs [n, out]"""
    obs_mean: torch.Tensor
    obs_inv_std: torch.Tensor
    obs_clip: float
    action_dim: int
    act: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
    """(outputs [..., out], base noise [..., d], scale) -> physical actions [..., d]"""
    base_noise: str = "normal"
    """distribution of the common base noise: normal or uniform"""
    subsets: dict[str, Callable[[str, torch.Tensor], torch.Tensor]] = field(default_factory=dict)
    """name -> (parameter name, parameter) -> 0/1 mask of the entries a direction may move"""
    critic: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    """(normalized observations [n, obs], physical actions [n, d]) -> Q [n], unperturbed"""
    regularizer: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None
    """(outputs [..., out], base noise [..., d], scale) -> per-sample term the actor adds to Q (e.g. -T log pi),
    so the regularized surrogate is the objective the actor actually ascends"""

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return ((raw - self.obs_mean) * self.obs_inv_std).clamp(-self.obs_clip, self.obs_clip)

    def draw_noise(self, generator: torch.Generator, shape) -> torch.Tensor:
        device = generator.device
        if self.base_noise == "normal":
            return torch.randn(shape, generator=generator, device=device)
        if self.base_noise == "uniform":
            # Open interval: inverse CDFs are finite.
            return torch.rand(shape, generator=generator, device=device).clamp(1e-6, 1.0 - 1e-6)
        raise ValueError(f"unknown base noise {self.base_noise}")


# ---------------------------------------------------------------- directions

def base_parameters(module: nn.Module) -> Params:
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def dot(a: Params, b: Params) -> torch.Tensor:
    return sum((a[name] * b[name]).sum() for name in a)


def norm(direction: Params) -> torch.Tensor:
    return dot(direction, direction).sqrt()


def scaled(direction: Params, factor) -> Params:
    return {name: value * factor for name, value in direction.items()}


def difference(target: Params, base: Params) -> Params:
    return {name: target[name] - base[name] for name in base}


def restrict(direction: Params, mask: Params | None) -> Params:
    return direction if mask is None else {name: value * mask[name] for name, value in direction.items()}


def subset_mask(adapter: Adapter, base: Params, subset: str) -> Params | None:
    if subset == "all":
        return None
    if subset not in adapter.subsets:
        raise ValueError(f"unknown subset {subset}; available: all, {', '.join(adapter.subsets)}")
    rule = adapter.subsets[subset]
    return {name: rule(name, value).to(value.dtype) for name, value in base.items()}


def filter_normalize(direction: Params, base: Params) -> Params:
    """Li et al. (2018): each filter (row of a weight matrix) of the direction gets its weight row's norm;
    1-D parameters (biases, norm gains) are left out, so a random slice moves the filters only."""
    out = {}
    for name, value in direction.items():
        weight = base[name]
        if weight.dim() < 2:
            out[name] = torch.zeros_like(value)
            continue
        rows = value.flatten(1)
        target = weight.flatten(1).norm(dim=1, keepdim=True)
        out[name] = (rows * target / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)).view_as(value)
    return out


def random_direction(base: Params, seed: int, mask: Params | None = None) -> Params:
    generator = torch.Generator().manual_seed(seed)
    raw = {name: torch.randn(value.shape, generator=generator).to(value.device) for name, value in base.items()}
    # Masking before normalizing keeps the masked rows at their own weights' scale.
    return filter_normalize(restrict(raw, mask), base)


def orthogonalize(direction: Params, against: Params) -> Params:
    coefficient = dot(direction, against) / dot(against, against).clamp_min(1e-30)
    return {name: direction[name] - coefficient * against[name] for name in direction}


def grid_points(base: Params, directions: list[Params], coefficients: torch.Tensor) -> Params:
    """Stacked parameters base + sum_k c[p, k] d_k for every row p of coefficients [P, K]."""
    coefficients = coefficients.to(next(iter(base.values())).device, torch.float32)
    return {name: value[None] + torch.einsum("pk,k...->p...", coefficients,
                                            torch.stack([d[name] for d in directions]))
            for name, value in base.items()}


def take(points: Params, index) -> Params:
    return {name: value[index] for name, value in points.items()}


# ---------------------------------------------------------------- evaluation

_UNBATCHABLE: set[type] = set()


def batched_outputs(module: nn.Module, points: Params, observations: torch.Tensor) -> torch.Tensor:
    """outputs [P, n, out] of P parameter sets on their own observations [P, n, obs]."""
    def one(params, obs):
        return functional_call(module, params, (obs,))
    if type(module) not in _UNBATCHABLE:
        try:
            return vmap(one)(points, observations)
        except NotImplementedError:  # an op without a batching rule: loop from now on
            _UNBATCHABLE.add(type(module))
    return torch.stack([one(take(points, p), observations[p]) for p in range(observations.shape[0])])


@dataclass
class Rollouts:
    returns: np.ndarray
    """[P, episodes] undiscounted return of each point's first episode per seed"""
    lengths: np.ndarray
    states: torch.Tensor | None
    """normalized observations visited by the recorded point, [states, obs]"""


@torch.no_grad()
def episode_returns(make_env: Callable[[int], object], adapter: Adapter, points: Params, episodes: int,
                    scale, seed: int, max_envs: int = 512, record_point: int | None = None,
                    record_every: int = 8) -> Rollouts:
    """First-episode returns of every (point, episode) pair under common random numbers. ``scale`` is one
    noise amplitude for all points or one per point."""
    num_points = next(iter(points.values())).shape[0]
    device = adapter.obs_mean.device
    scales = torch.as_tensor(scale, dtype=torch.float32, device=device).expand(num_points)
    per_chunk = max(1, max_envs // episodes)
    returns = np.zeros((num_points, episodes))
    lengths = np.zeros((num_points, episodes), dtype=np.int64)
    recorded = []
    for start in range(0, num_points, per_chunk):
        stop = min(num_points, start + per_chunk)
        chunk = take(points, slice(start, stop))
        count = stop - start
        envs = make_env(count * episodes)
        try:
            raw, _ = envs.reset(seed=[seed + e for _ in range(count) for e in range(episodes)])
            generator = torch.Generator(device=device).manual_seed(seed)
            total = np.zeros(count * episodes)
            steps = np.zeros(count * episodes, dtype=np.int64)
            live = np.ones(count * episodes, dtype=bool)
            t = 0
            while live.any():
                obs = adapter.normalize(torch.as_tensor(raw, dtype=torch.float32, device=device))
                obs = obs.view(count, episodes, -1)
                if record_point is not None and start <= record_point < stop and t % record_every == 0:
                    recorded.append(obs[record_point - start][torch.as_tensor(live.reshape(count, episodes)[
                        record_point - start], device=device)].clone())
                noise = adapter.draw_noise(generator, (episodes, adapter.action_dim))
                actions = adapter.act(batched_outputs(adapter.module, chunk, obs), noise[None],
                                      scales[start:stop, None, None])
                actions = actions.reshape(count * episodes, adapter.action_dim)
                raw, reward, terminated, truncated, _ = envs.step(actions.cpu().numpy().astype(np.float32))
                total += np.where(live, reward, 0.0)
                steps += live
                live &= ~(np.asarray(terminated) | np.asarray(truncated))
                t += 1
            returns[start:stop] = total.reshape(count, episodes)
            lengths[start:stop] = steps.reshape(count, episodes)
        finally:
            envs.close()
    states = torch.cat(recorded) if recorded else None
    return Rollouts(returns, lengths, states)


def critic_objective(adapter: Adapter, points: Params, states: torch.Tensor, samples: int, scale: float,
                     seed: int, chunk: int = 16, regularized: bool = False) -> torch.Tensor:
    """E_s E_noise Q(s, a_theta(s, noise)) at every point with common noise, plus the adapter's regularizer
    when ``regularized``: the surrogate a critic-climbing actor ascends. Differentiable in points."""
    if adapter.critic is None:
        raise ValueError("this adapter has no Q critic")
    generator = torch.Generator(device=states.device).manual_seed(seed)
    noise = adapter.draw_noise(generator, (samples, states.shape[0], adapter.action_dim))
    num_points = next(iter(points.values())).shape[0]
    values = []
    for start in range(0, num_points, chunk):
        part = take(points, slice(start, min(num_points, start + chunk)))
        count = next(iter(part.values())).shape[0]
        outputs = batched_outputs(adapter.module, part, states.expand(count, -1, -1))
        actions = adapter.act(outputs[:, None], noise[None], scale)  # [count, samples, S, d]
        q = adapter.critic(states.expand(count, samples, -1, -1).reshape(-1, states.shape[-1]),
                           actions.reshape(-1, adapter.action_dim)).view(count, -1)
        if regularized and adapter.regularizer is not None:
            q = q + adapter.regularizer(outputs[:, None], noise[None], scale).reshape(count, -1)
        values.append(q.mean(1))
    return torch.cat(values)


def critic_ascent_direction(adapter: Adapter, base: Params, states: torch.Tensor, samples: int, scale: float,
                            seed: int, mask: Params | None = None) -> Params:
    """Gradient of the regularized critic surrogate at the base parameters: the pathwise actor's ascent
    direction without its trust region."""
    params = {name: value.clone().requires_grad_(True) for name, value in base.items()}
    stacked = {name: value[None] for name, value in params.items()}
    with torch.enable_grad():
        objective = critic_objective(adapter, stacked, states, samples, scale, seed, regularized=True)[0]
        grads = torch.autograd.grad(objective, list(params.values()), allow_unused=True)
    direction = {name: (g if g is not None else torch.zeros_like(base[name])).detach()
                 for name, g in zip(params, grads)}
    return restrict(direction, mask)


# ---------------------------------------------------------------- statistics

def paired_summary(returns: np.ndarray, center: int) -> dict[str, np.ndarray]:
    """Per-point mean return with its standard error, and the paired difference to the center point."""
    episodes = returns.shape[1]
    diff = returns - returns[center][None]
    return {"mean": returns.mean(1), "se": returns.std(1, ddof=1) / np.sqrt(episodes),
            "diff": diff.mean(1), "diff_se": diff.std(1, ddof=1) / np.sqrt(episodes)}


def paired_quadratic(coefficients: np.ndarray, diff: np.ndarray, diff_se: np.ndarray, center: int
                     ) -> dict[str, float]:
    """Fit paired differences d(c) = b (c - c0) + k (c - c0)^2 / 2 (no intercept: d(c0) = 0 exactly), weighted
    by 1 / se^2 with se floored at a tenth of the median so one lucky point cannot dominate. Returns slope b and
    curvature k at c0, and the peak c0 - b / k (meaningful only when k < 0)."""
    keep = np.arange(len(coefficients)) != center
    x = coefficients[keep] - coefficients[center]
    se = diff_se[keep]
    se = np.maximum(se, 0.1 * np.median(se) + 1e-12)
    design = np.stack([x, 0.5 * x ** 2], axis=1) / se[:, None]
    (b, k), *_ = np.linalg.lstsq(design, diff[keep] / se, rcond=None)
    covariance = np.linalg.pinv(design.T @ design)
    return {"slope": float(b), "slope_se": float(np.sqrt(covariance[0, 0])), "curvature": float(k),
            "curvature_se": float(np.sqrt(covariance[1, 1])),
            "peak": float(coefficients[center] - b / k) if k < 0 else float("nan")}


def quadratic_peak(coefficients: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None
                   ) -> dict[str, float]:
    """Weighted quadratic fit v = a + b c + k c^2 / 2: slope b and curvature k at 0, and the peak -b / k
    (meaningful only when k < 0)."""
    design = np.stack([np.ones_like(coefficients), coefficients, 0.5 * coefficients ** 2], axis=1)
    w = np.ones_like(values) if weights is None else weights
    solution, *_ = np.linalg.lstsq(design * np.sqrt(w)[:, None], values * np.sqrt(w), rcond=None)
    a, b, k = solution
    return {"value": float(a), "slope": float(b), "curvature": float(k),
            "peak": float(-b / k) if k < 0 else float("nan")}
