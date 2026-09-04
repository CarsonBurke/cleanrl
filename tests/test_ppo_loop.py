"""Tests for the shared PPO rollout/update-loop primitives."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache,
    compute_gae,
    device_minibatches,
    explained_variance,
    gather_metrics,
    get_gae_fn,
)

ROOT = Path(__file__).parents[1]


def _load_baseline():
    spec = importlib.util.spec_from_file_location(
        "ppo_continuous_action_baseline", ROOT / "cleanrl/ppo_continuous_action.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _random_gae_inputs(time_steps=256, num_envs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.rand(shape, generator=g, dtype=torch.float32)
    rewards = rand(time_steps, num_envs)
    values = rand(time_steps, num_envs)
    terminations = (rand(time_steps, num_envs) < 0.03).float()
    truncations = (rand(time_steps, num_envs) < 0.05).float()
    truncations[terminations.bool()] = 0.0  # no double boundaries
    bootstrap = rand(time_steps, num_envs)
    tail = rand(num_envs)
    return rewards, values, terminations, truncations, bootstrap, tail


def test_gae_matches_baseline():
    baseline = _load_baseline().compute_gae
    args = _random_gae_inputs() + (0.99, 0.95)
    expected_adv, expected_ret = baseline(*args)
    adv, ret = compute_gae(*args)
    torch.testing.assert_close(adv, expected_adv, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ret, expected_ret, atol=1e-5, rtol=1e-5)
    # Eager factory returns the plain function (cached).
    assert get_gae_fn(compiled=False) is compute_gae
    assert get_gae_fn(compiled=False) is get_gae_fn(compiled=False)


def test_gae_terminal_truncation_semantics():
    # One env, two steps: termination at t=0 must kill bootstrap AND trace;
    # truncation at t=0 must keep bootstrap but kill trace.
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.5]])
    bootstrap = torch.tensor([[10.0], [10.0]])
    tail = torch.tensor([0.0])
    adv_term, _ = compute_gae(
        rewards, values,
        torch.tensor([[1.0], [0.0]]), torch.tensor([[0.0], [0.0]]),
        bootstrap, tail, 0.99, 0.95,
    )
    # delta = 1 + 0 - 0.5; trace cut -> advantage == delta.
    assert adv_term[0, 0].item() == pytest.approx(0.5)
    adv_trunc, _ = compute_gae(
        rewards, values,
        torch.tensor([[0.0], [0.0]]), torch.tensor([[1.0], [0.0]]),
        bootstrap, tail, 0.99, 0.95,
    )
    # delta = 1 + 0.99*10 - 0.5; trace cut -> advantage == delta.
    assert adv_trunc[0, 0].item() == pytest.approx(1.0 + 9.9 - 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gae_compiled_matches_eager_on_cuda(capsys):
    args = [a.cuda() for a in _random_gae_inputs(time_steps=64, num_envs=4)]
    expected = compute_gae(*args, 0.99, 0.95)
    compiled = get_gae_fn(compiled=True)
    t = time.perf_counter()
    got = compiled(*args, 0.99, 0.95)
    elapsed = time.perf_counter() - t
    torch.testing.assert_close(got[0], expected[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got[1], expected[1], atol=1e-4, rtol=1e-4)
    with capsys.disabled():
        print(f"\n[ppo_loop] compiled GAE (64x4, incl. compile): {elapsed:.2f}s")


def test_truncation_cache_matches_per_step():
    torch.manual_seed(0)
    steps, num_envs, dim = 32, 6, 5
    value_layer = nn.Linear(dim, 1)
    rng = np.random.default_rng(0)
    truncs = rng.random((steps, num_envs)) < 0.2
    finals = rng.normal(size=(steps, num_envs, dim))
    cache = TruncationBootstrapCache(steps, num_envs)
    expected = torch.zeros(steps, num_envs)
    with torch.no_grad():
        for s in range(steps):
            infos = {
                "final_observation": [
                    finals[s, i].copy() if truncs[s, i] else None for i in range(num_envs)
                ],
                "_final_observation": truncs[s].copy(),
            }
            cache.push(s, truncs[s], infos)
            for i in np.flatnonzero(truncs[s]):
                obs = torch.as_tensor(finals[s, i], dtype=torch.float32)
                expected[s, i] = value_layer(obs).flatten()[0]
        got = cache.resolve(lambda b: value_layer(b), torch.device("cpu"))
    assert len(cache) == int(truncs.sum())
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_truncation_cache_empty_never_calls_value_fn():
    cache = TruncationBootstrapCache(8, 2)

    def explode(batch):
        raise AssertionError("must not be called")

    out = cache.resolve(explode, torch.device("cpu"))
    assert out.shape == (8, 2)
    assert bool((out == 0).all())


def test_device_minibatches_cover_permutation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(1)
    batches = device_minibatches(100, 32, device, generator=gen)
    assert [len(b) for b in batches] == [32, 32, 32, 4]
    assert all(b.device.type == device.type and b.dtype == torch.int64 for b in batches)
    seen = torch.cat(batches).cpu().sort().values
    torch.testing.assert_close(seen, torch.arange(100))


def test_gather_metrics_matches_individual_items():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    named = {
        "clipfrac": torch.tensor(0.25, device=device, requires_grad=True),
        "approx_kl": torch.tensor(0.01, dtype=torch.float64, device=device),
        "entropy": torch.tensor(1.5, device=device) * 2,  # non-leaf graph output
    }
    got = gather_metrics(named)
    assert list(got.keys()) == ["clipfrac", "approx_kl", "entropy"]
    assert got["clipfrac"] == pytest.approx(0.25)
    assert got["approx_kl"] == pytest.approx(0.01)
    assert got["entropy"] == pytest.approx(3.0)
    assert all(isinstance(v, float) for v in got.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gather_metrics_mixed_cpu_cuda():
    # Regression test: CPU-computed scalars (e.g. from CPU-resident critic
    # targets) must not break the single stacked transfer.
    got = gather_metrics(
        {
            "cuda_loss": torch.tensor(0.5, device="cuda"),
            "cpu_scalar": torch.tensor(0.25),
        }
    )
    assert got["cuda_loss"] == pytest.approx(0.5)
    assert got["cpu_scalar"] == pytest.approx(0.25)


def test_explained_variance_matches_numpy():
    rng = np.random.default_rng(0)
    pred = torch.as_tensor(rng.normal(size=1000), dtype=torch.float32)
    true = torch.as_tensor(rng.normal(size=1000), dtype=torch.float32)
    got = explained_variance(pred, true)
    assert got.shape == ()
    assert got.item() == pytest.approx(
        1 - np.var(true.numpy() - pred.numpy()) / np.var(true.numpy())
    )
    nan = explained_variance(torch.ones(16), torch.ones(16))
    assert nan.shape == () and bool(torch.isnan(nan))
