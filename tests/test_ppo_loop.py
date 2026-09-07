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
    compute_gae_from_next_values,
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
@pytest.mark.cuda
def test_gae_compiled_matches_eager_on_cuda(capsys):
    args = [a.cuda() for a in _random_gae_inputs(time_steps=64, num_envs=4)]
    expected = compute_gae(*args, 0.99, 0.95)
    compiled = get_gae_fn(compiled=True)
    t = time.perf_counter()
    got = compiled(*args, 0.99, 0.95)
    elapsed = time.perf_counter() - t
    torch.testing.assert_close(got[0], expected[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[1], expected[1], atol=1e-5, rtol=1e-5)
    with capsys.disabled():
        print(f"\n[ppo_loop] compiled GAE (64x4, incl. compile): {elapsed:.2f}s")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
def test_truncation_cache_matches_per_step():
    torch.manual_seed(0)
    steps, num_envs, dim = 32, 6, 5
    value_layer = nn.Linear(dim, 1, device="cuda")
    rng = np.random.default_rng(0)
    truncs = rng.random((steps, num_envs)) < 0.2
    finals = rng.normal(size=(steps, num_envs, dim))
    cache = TruncationBootstrapCache(steps, num_envs)
    expected = torch.zeros(steps, num_envs, device="cuda")
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
                obs = torch.as_tensor(finals[s, i], dtype=torch.float32, device="cuda")
                expected[s, i] = value_layer(obs).flatten()[0]
        got = cache.resolve(lambda b: value_layer(b), torch.device("cuda"))
    assert len(cache) == int(truncs.sum())
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_truncation_cache_empty_never_calls_value_fn():
    cache = TruncationBootstrapCache(8, 2)

    def explode(batch):
        raise AssertionError("must not be called")

    out = cache.resolve(explode, torch.device("cpu"))
    assert out.shape == (8, 2)
    assert bool((out == 0).all())


def test_truncation_cache_snapshots_mutable_final_observations():
    cache = TruncationBootstrapCache(3, 2)
    mutable = np.array([2.0, 5.0])
    cache.push(0, [True, False], {"final_observation": [mutable, None]})
    mutable[:] = -100
    out = cache.resolve(lambda batch: batch.sum(-1), "cpu")
    torch.testing.assert_close(out, torch.tensor([[7.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))


def test_truncation_cache_normalized_rows_reset_and_fixed_batches():
    cache = TruncationBootstrapCache(3, 2, obs_shape=(2,))
    transitions = np.array([[2.0, 5.0], [3.0, 8.0]], dtype=np.float64)
    cache.push_normalized(0, [True, False], transitions)
    cache.push_normalized(1, [True, True], transitions)
    transitions[:] = -1
    shapes = []

    def value_fn(batch):
        shapes.append(tuple(batch.shape))
        return batch.sum(-1, keepdim=True)

    got = cache.resolve(value_fn, "cpu", batch_size=2)
    assert shapes == [(2, 2), (2, 2)]
    torch.testing.assert_close(got, torch.tensor([[7.0, 0.0], [7.0, 11.0], [0.0, 0.0]]))
    assert len(cache) == 3
    storage = cache._observations
    cache.reset()
    assert len(cache) == 0 and cache._observations is storage
    cache.push_normalized(2, [False, True], transitions)
    got = cache.resolve(value_fn, "cpu", batch_size=2)
    torch.testing.assert_close(got, torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, -2.0]]))


def test_truncation_cache_duplicate_slot_keeps_latest_snapshot():
    cache = TruncationBootstrapCache(1, 2)
    cache.push_normalized(0, [True, False], np.array([[1.0], [2.0]]))
    cache.push_normalized(0, [True, True], np.array([[3.0], [4.0]]))
    got = cache.resolve(lambda batch: batch.to(torch.bfloat16), "cpu")
    torch.testing.assert_close(got, torch.tensor([[3.0, 4.0]]))


@pytest.mark.parametrize("method", ["push_normalized", "push"])
def test_truncation_cache_validates_arguments_when_nothing_truncated(method):
    # The no-truncation fast path returns before building any index array, so
    # it must still reject a step outside the rollout and a truncations vector
    # of the wrong width instead of silently recording nothing.
    cache = TruncationBootstrapCache(4, 3, obs_shape=(2,))
    payload = (np.zeros((3, 2), dtype=np.float32) if method == "push_normalized"
               else {"final_observation": [None, None, None]})
    call = getattr(cache, method)
    with pytest.raises(ValueError, match=r"truncations must have shape \(3,\)"):
        call(0, [False, False], payload)
    with pytest.raises(IndexError, match=r"outside \[0, 4\)"):
        call(4, [False, False, False], payload)
    call(0, [False, False, False], payload)
    assert len(cache) == 0


def test_truncation_cache_observation_storage_scales_with_truncations():
    steps, envs, dim = 2048, 64, 376
    cache = TruncationBootstrapCache(steps, envs, obs_shape=(dim,))
    assert cache._observations.nbytes == envs * dim * 4
    transitions = np.ones((envs, dim), dtype=np.float32)
    cache.push_normalized(steps - 1, np.ones(envs, dtype=bool), transitions)
    assert cache._observations.nbytes == envs * dim * 4
    cache.push_normalized(0, np.ones(envs, dtype=bool), 2 * transitions)
    assert cache._observations.nbytes == 2 * envs * dim * 4
    got = cache.resolve(lambda batch: batch.sum(-1), "cpu")
    torch.testing.assert_close(got[0], torch.full((envs,), 2.0 * dim))
    torch.testing.assert_close(got[-1], torch.full((envs,), float(dim)))
    assert torch.count_nonzero(got) == 2 * envs
    storage = cache._observations
    cache.reset()
    cache.push_normalized(steps - 1, np.ones(envs, dtype=bool), 3 * transitions)
    assert cache._observations is storage
    assert len(cache) == envs
    got = cache.resolve(lambda batch: batch.sum(-1), "cpu")
    assert torch.count_nonzero(got) == envs
    torch.testing.assert_close(got[-1], torch.full((envs,), 3.0 * dim))


def test_truncation_cache_grows_to_all_slots_without_losing_prior_rows():
    cache = TruncationBootstrapCache(5, 3)
    for step in reversed(range(5)):
        transitions = np.arange(3, dtype=np.float32)[:, None] + 3 * step
        cache.push_normalized(step, np.ones(3, dtype=bool), transitions)
    assert len(cache) == 15
    assert cache._capacity == 15
    got = cache.resolve(lambda batch: batch.flatten(), "cpu", batch_size=4)
    torch.testing.assert_close(got, torch.arange(15, dtype=torch.float32).reshape(5, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
def test_truncation_cache_compiled_fixed_batches_cuda():
    cache = TruncationBootstrapCache(4, 3, obs_shape=(2,))

    def value_fn(batch):
        return batch.square().sum(-1)

    compiled = torch.compile(value_fn, mode="reduce-overhead")
    for rollout in range(3):
        cache.reset()
        expected = np.zeros((4, 3), dtype=np.float32)
        for step in range(4):
            observations = np.arange(6, dtype=np.float32).reshape(3, 2) + step + rollout
            truncations = np.array([True, step % 2 == 0, True])
            cache.push_normalized(step, truncations, observations)
            expected[step, truncations] = np.square(observations[truncations]).sum(-1)
        with torch.no_grad():
            # Several calls per resolve and several resolves reuse the compiled
            # value output storage; each scatter must consume it before reuse.
            got = cache.resolve(compiled, "cuda", batch_size=3)
        torch.testing.assert_close(got.cpu(), torch.from_numpy(expected), atol=0, rtol=0)


@pytest.mark.parametrize(
    "infos",
    [
        {},
        {"final_observation": [None, None]},
        {"final_observation": [np.ones(2), None], "_final_observation": [False, False]},
    ],
)
def test_truncation_cache_rejects_missing_final(infos):
    with pytest.raises(RuntimeError, match="final.observation"):
        TruncationBootstrapCache(2, 2).push(0, [True, False], infos)


def test_explicit_next_values_gae_matches_v30_recurrence():
    rewards, values, terms, truncs, next_values, _ = _random_gae_inputs(31, 7)
    terms[2, 3] = truncs[2, 3] = 1.0
    gamma, gae_lambda = 0.99, 0.95
    expected = torch.empty_like(rewards)
    running = torch.zeros_like(next_values[-1])
    boundaries = torch.logical_or(terms, truncs)
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_values[step] * (1.0 - terms[step]) - values[step]
        running = torch.where(boundaries[step], delta, delta + gamma * gae_lambda * running)
        expected[step] = running
    got, returns = compute_gae_from_next_values(
        rewards, values, terms, truncs, next_values, gamma, gae_lambda
    )
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    torch.testing.assert_close(returns, expected + values, atol=0, rtol=0)
    assert get_gae_fn(explicit_next_values=True) is compute_gae_from_next_values
    assert get_gae_fn() is compute_gae


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
def test_explicit_next_values_gae_compiled_cuda():
    rewards, values, terms, truncs, next_values, _ = [
        tensor.cuda() for tensor in _random_gae_inputs(64, 4)
    ]
    args = (rewards, values, terms, truncs, next_values, 0.99, 0.95)
    expected = compute_gae_from_next_values(*args)
    got = get_gae_fn(compiled=True, explicit_next_values=True)(*args)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
@pytest.mark.parametrize("explicit", [False, True])
def test_compiled_gae_tile_boundaries_and_nonfinite_trace(explicit):
    inputs = [tensor.cuda() for tensor in _random_gae_inputs(67, 3, seed=1)]
    rewards, values, terms, truncs, next_values, tail = inputs
    terms.zero_()
    truncs.zero_()
    terms[[0, 2, 31, 32, 63, 66], 0] = 1
    truncs[[1, 3, 30, 33, 64, 66], 1] = 1
    terms[32, 2] = truncs[32, 2] = 1
    # Fractional masks intentionally remain numerical masks, not bool traces.
    terms[17, 2], truncs[18, 2] = 0.25, 0.5
    # Explicit where must cut a nonfinite future; standard multiply must not.
    rewards[65, 1] = float("nan")
    truncs[64, 1] = 1
    args = (*inputs[:5], 0.99, 0.95) if explicit else (*inputs, 0.99, 0.95)
    reference = compute_gae_from_next_values if explicit else compute_gae
    expected = reference(*args)
    rng = torch.cuda.get_rng_state()
    torch.compiler.cudagraph_mark_step_begin()
    actual = get_gae_fn(compiled=True, explicit_next_values=explicit)(*args)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert bool(actual[0][64, 1].isfinite()) == explicit


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("case", ["strided", "float64", "mixed", "autograd"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")])])
def test_compiled_gae_preserves_reference_input_contract(explicit, case, device):
    inputs = [tensor.to(device) for tensor in _random_gae_inputs(5, 3, seed=1)]
    if case == "strided":
        inputs = [
            tensor.t().contiguous().t() if tensor.ndim == 2 else tensor
            for tensor in inputs
        ]
    elif case == "float64":
        inputs = [tensor.double() for tensor in inputs]
    elif case == "mixed":
        inputs[1] = inputs[1].double()
        inputs[2] = inputs[2].to(torch.int32)
    else:
        for index in (0, 1, 4, 5):
            inputs[index].requires_grad_()
    args = (*inputs[:5], 0.99, 0.95) if explicit else (*inputs, 0.99, 0.95)
    reference = compute_gae_from_next_values if explicit else compute_gae
    expected = reference(*args)
    torch.compiler.cudagraph_mark_step_begin()
    actual = get_gae_fn(compiled=True, mode="default", explicit_next_values=explicit)(*args)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if case == "autograd":
        differentiated = [inputs[index] for index in ((0, 1, 4) if explicit else (0, 1, 4, 5))]
        weights = torch.arange(15, device=device).reshape(5, 3) / 15
        expected_gradients = torch.autograd.grad(
            (expected[0] * weights + expected[1]).sum(), differentiated
        )
        actual_gradients = torch.autograd.grad(
            (actual[0] * weights + actual[1]).sum(), differentiated
        )
        torch.testing.assert_close(actual_gradients, expected_gradients, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")])])
def test_device_minibatches_cover_permutation(device):
    device = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(1)
    batches = device_minibatches(100, 32, device, generator=gen)
    assert [len(b) for b in batches] == [32, 32, 32, 4]
    assert all(b.device.type == device.type and b.dtype == torch.int64 for b in batches)
    seen = torch.cat(batches).cpu().sort().values
    torch.testing.assert_close(seen, torch.arange(100))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")])])
def test_gather_metrics_matches_individual_items(device):
    device = torch.device(device)
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
@pytest.mark.cuda
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
