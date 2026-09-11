"""Fused optimizer behavior against the independent CUDA FP32 v2 eager oracle.

GPU execution must be queued via mlq; neither compilation nor capture is mocked.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from cleanrl.plasticity import predictive_transport_stream_v2 as reference
from cleanrl.plasticity import predictive_transport_stream_v3 as transport
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

GRID = tuple((lr, beta) for lr in (0.007, 0.025) for beta in (0.9, 0.99999))
STATE_LISTS = ("weights", "previous_weights", "m", "v")
CANDIDATE_STATE = ("error", "correction_norm_sum", "finite_candidates")


def learner(method="predictive", grid=GRID, implementation=transport):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    initial = [torch.tensor(value, device="cuda", dtype=torch.float32) for value in (
        [[0.7, -0.2, 0.1], [-0.3, 0.6, -0.2]],
        [[0.8, -0.4, -0.1], [0.2, 0.5, 0.3]],
        [[0.4, -0.5, 0.2]])]
    base_x = torch.tensor([[-1.2, 0.3], [0.4, -0.8], [1.1, 0.7], [-0.2, -0.6],
                           [0.8, 0.1], [-0.7, 1.0], [0.3, 0.9], [-0.4, 0.2]], device="cuda")
    base_y = torch.tensor([0.7, -0.4, 1.2, 0.9, -0.8, 0.5, -0.7, 0.3], device="cuda")
    # Eight distinct blocks ensure a repeated/skipped observation changes the trajectory.
    xs = torch.cat([base_x * (1 + block * 0.07) + block * 0.03 for block in range(8)])
    ys = torch.cat([base_y + block * 0.04 for block in range(8)])
    return implementation.Learner(method, grid, initial, SimpleNamespace(graph_steps=16),
                                  xs, ys, ys * 0.6, torch.ones_like(ys))


def assert_learning_state(actual, expected, *, candidate=None, rtol=3e-3, atol=3e-5):
    for field in STATE_LISTS:
        for observed, wanted in zip(getattr(actual, field), getattr(expected, field)):
            if candidate is not None:
                observed = observed[candidate:candidate + 1]
            torch.testing.assert_close(observed, wanted, rtol=rtol, atol=atol)
    for field in CANDIDATE_STATE:
        observed, wanted = getattr(actual, field), getattr(expected, field)
        if candidate is not None:
            observed = observed[candidate:candidate + 1]
        torch.testing.assert_close(observed, wanted, rtol=rtol, atol=atol)
    indices = actual.candidate_indices if candidate is None else actual.candidate_indices[candidate:candidate + 1]
    torch.testing.assert_close(indices, expected.index.expand_as(indices), rtol=0, atol=0)
    if candidate is None:
        for field in ("index", "steps", "null_error"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=rtol, atol=atol)


def assert_snapshot(model, expected):
    for observed, wanted in zip(model.mutable, expected):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_five_arm_trajectory_matches_independent_v2_eager(method):
    actual = learner(method)
    eager = learner(method, implementation=reference)
    for step in range(1, 33):
        actual.update()
        eager.update()
        assert_learning_state(actual, eager)
        assert actual.index.item() == actual.steps.item() == step
    assert all(value.dtype == torch.float32 for value in actual.mutable if value.is_floating_point())


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_candidate_owned_indices_consume_independent_observations_and_moment_masses(method):
    batch = learner(method, (GRID[0], GRID[-1]))
    singles = [learner(method, (config,), implementation=reference) for config in batch.grid]
    # Candidate one starts three samples ahead with matching nonzero model/moment history.
    for _ in range(3):
        singles[1].update()
    for field in STATE_LISTS:
        for target, source in zip(getattr(batch, field), getattr(singles[1], field)):
            target[1:2].copy_(source)
    for field in CANDIDATE_STATE:
        getattr(batch, field)[1:2].copy_(getattr(singles[1], field))
    batch.candidate_indices[1].copy_(singles[1].index)
    batch._launch(16)
    for candidate, single in enumerate(singles):
        for _ in range(16):
            single.update()
        assert_learning_state(batch, single, candidate=candidate)
    assert batch.candidate_indices.tolist() == [16, 19]
    assert batch.index.item() == batch.steps.item() == 16
    torch.testing.assert_close(batch.null_error, singles[0].null_error, rtol=3e-3, atol=3e-5)


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_fused_multistep_consumes_all_samples_and_preserves_previous_weight_history(method):
    fused, singles = learner(method), learner(method)
    eager = learner(method, implementation=reference)
    for block in range(2):
        fused._launch(16)
        for _ in range(16):
            singles.update()
            eager.update()
        assert_learning_state(fused, eager)
        assert_learning_state(fused, singles)
        assert fused.candidate_indices.tolist() == [16 * (block + 1)] * len(GRID)
        assert fused.steps.item() == 16 * (block + 1)


@pytest.mark.parametrize("method", transport.METHODS)
@pytest.mark.parametrize("prefix", [0, 3], ids=["initial", "nonzero"])
@torch.no_grad()
def test_capture_rolls_back_and_replays_exact_fused_production_state(method, prefix):
    captured, fused = learner(method), learner(method)
    eager = learner(method, implementation=reference)
    for _ in range(prefix):
        captured.update()
        fused.update()
        eager.update()
    before = captured.snapshot()
    graph, parity = captured.capture()
    try:
        assert math.isfinite(parity)
        assert captured.capture_failed_candidates == [False] * len(GRID)
        assert_snapshot(captured, before)
        for _ in range(2):
            graph.replay()
            fused._launch(16)
            for _ in range(16):
                eager.update()
        torch.cuda.synchronize()
        assert_learning_state(captured, fused, rtol=0, atol=0)
        assert_learning_state(captured, eager)
        assert captured.candidate_indices.tolist() == [prefix + 32] * len(GRID)
        final = captured.snapshot()
        captured.restore(before)
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        assert_snapshot(captured, final)
    finally:
        del graph


@pytest.mark.parametrize("method", transport.METHODS)
@torch.no_grad()
def test_nonfinite_candidate_capture_retains_finite_partner_and_its_trajectory(method):
    batch = learner(method, (GRID[0], GRID[-1]))
    single = learner(method, (GRID[-1],), implementation=reference)
    batch.weights[0][0, 0, 0] = float("nan")
    before = batch.snapshot()
    graph, parity = batch.capture()
    try:
        assert math.isfinite(parity)
        assert batch.capture_failed_candidates == [True, False]
        assert_snapshot(batch, before)
        for _ in range(2):
            graph.replay()
            for _ in range(16):
                single.update()
        torch.cuda.synchronize()
        assert batch.finite_candidates.tolist() == [False, True]
        assert_learning_state(batch, single, candidate=1)
        assert batch.candidate_indices.tolist() == [32, 32]
    finally:
        del graph


@pytest.mark.parametrize("failure_side", ["fused", "eager"])
@pytest.mark.parametrize("failure_kind", ["validity", "finite_moment"])
@torch.no_grad()
def test_capture_rejects_one_sided_audit_mismatch_without_discarding_candidate(monkeypatch, failure_side, failure_kind):
    model = learner("predictive")
    for _ in range(3):
        model.update()
    before = model.snapshot()
    # Failure metadata is part of rollback, independently of the device validity buffer.
    model.capture_failed_candidates = [False, True, False, False]
    metadata_before = list(model.capture_failed_candidates)
    launch, eager_update = model._launch, reference.Learner.update
    calls = 0

    def inject_failure(target):
        if failure_kind == "validity":
            # Every weight/moment stays finite: validity disagreement alone must fail.
            target.finite_candidates[0] = False
        else:
            # A finite numeric parity failure must not become a dropped grid candidate.
            target.m[0][0, 0, 0].add_(1.0)

    def launch_with_fault(n_steps):
        nonlocal calls
        result = launch(n_steps)
        calls += 1
        if calls == 3 and failure_side == "fused":
            inject_failure(model)
        return result

    def eager_with_fault(self):
        result = eager_update(self)
        if failure_side == "eager":
            inject_failure(self)
        return result

    monkeypatch.setattr(model, "_launch", launch_with_fault)
    monkeypatch.setattr(reference.Learner, "update", eager_with_fault)
    with pytest.raises(AssertionError):
        model.capture()
    assert_snapshot(model, before)
    assert model.capture_failed_candidates == metadata_before



@pytest.mark.parametrize("prefix", [0, 3], ids=["initial", "nonzero"])
@pytest.mark.parametrize("failure_stage", ["warmup", "fused_block"])
@torch.no_grad()
def test_capture_exception_restores_state_and_failure_history(monkeypatch, prefix, failure_stage):
    model = learner("implicit")
    for _ in range(prefix):
        model.update()
    before = model.snapshot()
    model.capture_failed_candidates = [False, True, False, False]
    metadata_before = list(model.capture_failed_candidates)
    launch = model._launch

    def interrupted_launch(n_steps):
        result = launch(n_steps)
        if failure_stage == "warmup" or n_steps == model.capture_steps:
            model.capture_failed_candidates = [True] * len(GRID)
            raise RuntimeError("injected interruption after real kernel execution")
        return result

    monkeypatch.setattr(model, "_launch", interrupted_launch)
    with pytest.raises(RuntimeError, match="injected interruption"):
        model.capture()
    torch.cuda.synchronize()
    assert_snapshot(model, before)
    assert model.capture_failed_candidates == metadata_before