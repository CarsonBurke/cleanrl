"""Focused contracts only; CUDA tests must be queued through mlq, not learning runs."""

import numpy as np
import pytest
import torch

from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_predictive_memory_eval_v4 as evaluation
from cleanrl.plasticity.panel_predictive_memory_v4 import (
    CausalState, Config, GROUPS, Learner, REFERENCE_NAMES, RidgeReference,
    configurations, exp_logit_gradient,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contracts; queue with mlq")


def tiny_bank():
    b = panel_hd.Bank.__new__(panel_hd.Bank)
    b.T, b.N, b.L, b.F, b.cut = 80, 4, 32, 257, 48
    b.dev = torch.device("cuda")
    generator = torch.Generator(device=b.dev).manual_seed(71)
    b.zt = torch.randn((b.T, b.N + 1), device=b.dev, generator=generator)
    b.vst = torch.randn((b.T, b.N + 1), device=b.dev, generator=generator)
    b.cst, b.acst = b.zt[:, 1:].mean(1), b.zt[:, 1:].abs().mean(1)
    b.lag_idx = torch.arange(1, b.L + 1, device=b.dev)
    b.ones = torch.ones((b.N, 1), device=b.dev)
    b.mu = torch.tensor(1.3, device=b.dev)
    return b


def prime(b, valid, until=33):
    state = CausalState(b, valid)
    for t in range(until):
        state.observe(torch.tensor(t, device=b.dev))
    return state


@cuda
@torch.no_grad()
def test_old_exact_latest_current_and_memory_future_independence():
    b = tiny_bank()
    valid = torch.ones((b.T, b.N), dtype=torch.bool, device=b.dev)
    state = prime(b, valid, 41)
    t = torch.tensor(40, device=b.dev)
    old, latest, memory = (x.clone() for x in state.frames(t))
    assert torch.equal(old, b.feats(t))
    # A future channel perturbation cannot affect any frame or past memory.
    b.zt[41:].add_(40)
    b.vst[41:].sub_(30)
    b.cst[41:].add_(20)
    b.acst[41:].add_(10)
    rebuilt = prime(b, valid, 41)
    for actual, expected in zip(rebuilt.frames(t), (old, latest, memory)):
        assert torch.equal(actual, expected)
    # Current observed bar changes latest and memory, but not exact frozen old.
    b.zt[40, 1] += 2
    changed = prime(b, valid, 41).frames(t)
    assert torch.equal(changed[0], old)
    assert not torch.equal(changed[1], latest)
    assert not torch.equal(changed[2], memory)


@cuda
@torch.no_grad()
def test_reference_updates_use_current_validity_not_future_labels():
    b = tiny_bank()
    valid = torch.ones((b.T, b.N), dtype=torch.bool, device=b.dev)
    valid[40, 0] = False
    state = prime(b, valid, 40)
    before = state.own.clone()
    b.zt[40, 1] = 9
    state.observe(torch.tensor(40, device=b.dev))
    assert torch.equal(state.own[:, 0], before[:, 0])
    realized = b.zt[40, 2:].square().clamp_max(25).mean()
    previous = prime(b, valid, 40)
    expected = previous.cross + previous.reference_alpha * (realized - previous.cross)
    torch.testing.assert_close(state.cross, expected)
    # No target mask enters the state API. Entirely missing observations hold
    # references, while zero-filled feature memory still advances causally.
    valid[41] = False
    own, cross = state.own.clone(), state.cross.clone()
    state.observe(torch.tensor(41, device=b.dev))
    assert torch.equal(state.own, own) and torch.equal(state.cross, cross)


def prefix_raw(configs, bars=20):
    count = len(configs) + len(REFERENCE_NAMES)
    raw = np.zeros((bars, count, len(evaluation.METRICS)), dtype=np.float64)
    raw[:, :, 0] = np.arange(count)[None] + 1
    raw[:, :, 3:] = 4
    return raw


def test_prefix_locks_reject_suffix_and_nonfinite_candidates(tmp_path):
    configs = configurations()
    raw = prefix_raw(configs)
    raw[0, 0, 0] = np.inf
    locks = evaluation.select_locks(raw, configs, 10, 20, 30, 30)
    assert locks[GROUPS[0]]["column"] == 1
    assert not locks[GROUPS[0]]["candidates"][0]["eligible"]
    with pytest.raises(ValueError, match="prefix"):
        evaluation.select_locks(np.concatenate((raw, raw[:1])), configs, 10, 20, 30, 31)
    result = {"data_identity": {"cut": 30, "stream_start": 10}, "consumed_until_exclusive": 30,
              "source_sha256": {}, "prefix_health": [True] * raw.shape[1], "locks": locks}
    evaluation.persist_prefix_locks(tmp_path, raw, result)
    with pytest.raises(FileExistsError):
        evaluation.persist_prefix_locks(tmp_path, raw, result)
    result["consumed_until_exclusive"] = 31
    with pytest.raises(ValueError, match="suffix"):
        evaluation.persist_prefix_locks(tmp_path, raw, result)


def test_reference_authentication_uses_prefix_not_development_suffix(tmp_path):
    configs = configurations()
    raw = prefix_raw(configs, 25)
    identity = {"stream_start": 10, "selection_start": 20, "cut": 30}
    result = {"version": 4, "stage": "development", "view": "real", "status": "completed",
              "data_identity": identity, "source_sha256": {"model": "fixed"},
              "prefix_health": [True] * raw.shape[1], "consumed_until_exclusive": 30,
              "locks": evaluation.select_locks(raw[:20], configs, 10, 20, 30, 30)}
    evaluation.persist_prefix_locks(tmp_path, raw[:20], result)
    result["consumed_until_exclusive"] = 35
    result["artifacts"] = {"raw_metrics": "raw_metrics.npz"}

    def save():
        np.savez(tmp_path / "raw_metrics.npz", bar_index=np.arange(10, 35),
                 **{key: raw[:, :, i] for i, key in enumerate(evaluation.METRICS)})
        result["artifact_sha256"] = {"raw_metrics": evaluation.frozen.file_sha256(tmp_path / "raw_metrics.npz")}
        evaluation.frozen.save_json(tmp_path / "results.json", result)

    save()
    original, _ = evaluation.load_reference(tmp_path / "results.json", result["source_sha256"], configs)
    raw[20:, :, 0] = np.arange(raw.shape[1])[::-1] * 1e8
    save()
    changed, _ = evaluation.load_reference(tmp_path / "results.json", result["source_sha256"], configs)
    assert original == changed
    raw[0, 0, 0] += 1
    save()
    with pytest.raises(ValueError, match="provenance"):
        evaluation.load_reference(tmp_path / "results.json", result["source_sha256"], configs)


@pytest.mark.parametrize("qlike", [False, True])
def test_manual_exp_logit_gradient_matches_autograd(qlike):
    logit = torch.tensor([-4., -.4, 0., 2., 4.], dtype=torch.float64, requires_grad=True)
    raw = torch.tensor([0., .01, 1., 15., 25.], dtype=torch.float64)
    mean = logit.exp()
    loss = (logit + raw / mean).sum() if qlike else (mean - raw).square().sum()
    expected, = torch.autograd.grad(loss, logit)
    actual = exp_logit_gradient(mean.detach(), raw, torch.tensor(qlike))
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


@cuda
@torch.no_grad()
def test_exp_initial_function_matching_and_unfloored_failure():
    b = tiny_bank()
    configs = tuple(Config(g, .001) for g in GROUPS)
    model = Learner(b.F, 8, b.mu, configs, b.dev, bins=5, num_samples=b.N)
    old, latest, memory = model.groups[:3]
    assert torch.equal(old.weights[0], latest.weights[0])
    assert torch.equal(latest.weights[0], memory.weights[0][:, :, :b.F])
    assert torch.count_nonzero(memory.weights[0][:, :, b.F:]) == 0
    exp = model.groups[-1]
    for p in exp.parameters:
        assert torch.equal(p[0], p[1])
    x = torch.ones((b.N, b.F + 24), device=b.dev)
    mask = torch.ones(b.N, dtype=torch.bool, device=b.dev)
    prediction = exp.step(x, torch.zeros(b.N, device=b.dev), mask)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=1e-6, rtol=0)
    exp.biases[2].fill_(1000)
    exp.step(x, torch.zeros(b.N, device=b.dev), mask)
    assert not exp.healthy.any()
    assert not torch.isfinite(exp.prediction).all()


@cuda
def test_exp_full_manual_adam_matches_autograd_with_missing_labels():
    b = tiny_bank()
    model = Learner(b.F, 8, b.mu, tuple(Config(g, .001) for g in GROUPS), b.dev, bins=5, num_samples=b.N)
    exp = model.groups[-1]
    params = [p.detach().clone().requires_grad_() for p in exp.parameters]
    optim = torch.optim.Adam(params, lr=.001, foreach=False)
    x = b.feats(torch.tensor(40, device=b.dev))
    x = torch.cat((x, torch.ones((b.N, 24), device=b.dev)), dim=1)
    mask = torch.tensor([True, False, True, True], device=b.dev)
    y = torch.tensor([-.3, float("nan"), 2., 10.], device=b.dev)
    for _ in range(5):
        w1, b1, w2, b2, w3, b3 = params
        h1 = torch.tanh(torch.matmul(x, w1.transpose(1, 2)) + b1[:, None])
        h2 = torch.tanh(torch.bmm(h1, w2.transpose(1, 2)) + b2[:, None])
        logits = (torch.bmm(h2, w3.transpose(1, 2)) + b3[:, None]).squeeze(-1)
        mean = logits.exp()
        raw = torch.where(mask, y + b.mu, 0.)
        loss = torch.where(mask, (mean[0] - raw).square(), 0.).sum() / mask.sum()
        loss += torch.where(mask, logits[1] + raw / mean[1], 0.).sum() / mask.sum()
        optim.zero_grad()
        loss.backward()
        expected_prediction = (mean - b.mu).detach()
        actual = exp.step(x, y, mask)
        torch.testing.assert_close(actual, expected_prediction, atol=2e-6, rtol=2e-6)
        optim.step()
        for actual_parameter, expected_parameter in zip(exp.parameters, params):
            torch.testing.assert_close(actual_parameter, expected_parameter, atol=3e-6, rtol=3e-6)
    saved = [p.clone() for p in (*exp.parameters, *exp.first_moments, *exp.second_moments)]
    clock = exp.adam_steps.clone()
    exp.step(x, torch.full_like(y, float("nan")), torch.zeros_like(mask))
    assert torch.equal(clock, exp.adam_steps)
    for actual, expected in zip((*exp.parameters, *exp.first_moments, *exp.second_moments), saved):
        assert torch.equal(actual, expected)


@cuda
@torch.no_grad()
def test_fullgraph_restores_all_state_and_target_cannot_change_current_forecast(monkeypatch):
    b = tiny_bank()
    observed = torch.ones((b.T, b.N), dtype=torch.bool, device=b.dev)
    state = prime(b, observed)
    model = Learner(b.F, 8, b.mu, tuple(Config(g, .001) for g in GROUPS), b.dev, bins=5, num_samples=b.N)
    target = torch.full((b.T, b.N), -.3, device=b.dev)
    valid = observed.clone()
    valid[:, -1] = False
    target[:, -1] = float("nan")
    runner = evaluation.Runner(b, model, target, valid, state, 32, b.L + 1, b.T - 1)
    initial = [t.clone() for t in runner.mutable]
    graphs = runner.capture()
    for actual, expected in zip(runner.mutable, initial):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    graphs[1].replay()
    torch.cuda.synchronize()
    first = model.prediction.clone()
    assert (runner.raw[0, :, -1] == 3).all()
    for actual, expected in zip(runner.mutable, initial):
        actual.copy_(expected)
    target[runner.start, :-1] = 3.
    graphs[1].replay()
    torch.cuda.synchronize()
    assert torch.equal(model.prediction, first)
    assert torch.isfinite(runner.raw[0]).all()
    before_failure = [t.clone() for t in runner.mutable]

    def reject(*args, **kwargs):
        raise RuntimeError("injected graph failure")

    monkeypatch.setattr(torch.cuda, "graph", reject)
    with pytest.raises(RuntimeError, match="injected graph failure"):
        runner.capture()
    for actual, expected in zip(runner.mutable, before_failure):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))


@cuda
@torch.no_grad()
def test_ridge_prefix_fit_and_suffix_labels_do_not_change_frozen_coefficients():
    ridge = RidgeReference("cuda")
    refs = torch.tensor([[0., 0., 0.], [.1, .4, 1.], [.2, .5, 2.], [.3, .6, 3.],
                         [.7, .7, .7], [.8, .8, .8], [.9, .9, .9]], device="cuda")
    y = torch.tensor([1., float("nan"), 3.], device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    ridge.step(refs, y, mask, torch.tensor(True, device="cuda"))
    ridge.fit()
    coefficient = ridge.coefficient.clone()
    a = ridge.step(refs, y, mask, torch.tensor(False, device="cuda"))
    b = ridge.step(refs, y * 10, mask, torch.tensor(False, device="cuda"))
    assert torch.equal(a, b)
    assert torch.equal(ridge.coefficient, coefficient)
    assert torch.isfinite(a).all() and not torch.equal(a, torch.zeros_like(a))


def test_confirmation_and_null_cannot_self_select():
    with pytest.raises(ValueError, match="reference-result"):
        evaluation.validate_args(evaluation.Args(output_dir="x", stage="confirmation", rank_offset=400))
    with pytest.raises(ValueError, match="reference-result"):
        evaluation.validate_args(evaluation.Args(output_dir="x", view="permuted"))
    with pytest.raises(ValueError, match="rank400"):
        evaluation.validate_args(evaluation.Args(output_dir="x", stage="confirmation", reference_result="locked"))


@pytest.mark.parametrize("changed", [{"min_coverage": .8}, {"vol_span": 100}])
def test_confirmation_rejects_preprocessing_that_moves_historical_boundary(changed):
    with pytest.raises(ValueError, match="historical"):
        evaluation.validate_args(evaluation.Args(output_dir="x", stage="confirmation", rank_offset=400,
                                                reference_result="locked", **changed))


def test_progress_checkpoint_preserves_consumed_metric_rows_without_finalization(tmp_path):
    import json

    journal = np.lib.format.open_memmap(tmp_path / "metrics.npy", mode="w+", dtype=np.float64, shape=(5, 2, 5))
    first = np.arange(30, dtype=np.float64).reshape(3, 2, 5)
    journal[:3] = first
    result = {"status": "running", "checkpoint_raw_bounds": [10, 13]}
    evaluation.persist_progress(tmp_path, result, journal, None)
    saved = np.load(tmp_path / "metrics.npy", mmap_mode="r")
    np.testing.assert_array_equal(saved[:3], first)
    assert json.loads((tmp_path / "results.json").read_text())["checkpoint_raw_bounds"] == [10, 13]
    journal[3:] = 7
    result["checkpoint_raw_bounds"] = [10, 15]
    evaluation.persist_progress(tmp_path, result, journal, None)
    np.testing.assert_array_equal(saved[:3], first)
    np.testing.assert_array_equal(saved[3:], np.full((2, 2, 5), 7))
    assert json.loads((tmp_path / "results.json").read_text())["checkpoint_raw_bounds"] == [10, 15]


def test_confirmation_authenticates_cache_and_preprocessing_identity():
    import copy
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks/plasticity/panel_frontier_v3_evidence.json"
    historical = json.loads(path.read_text())
    identity = copy.deepcopy(historical["data_identity"])
    assert evaluation.authenticate_confirmation(identity, historical) == 90145
    identity["cache_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="cache_sha256"):
        evaluation.authenticate_confirmation(identity, historical)
    identity = copy.deepcopy(historical["data_identity"])
    identity["preprocessing"]["min_coverage"] = .8
    with pytest.raises(ValueError, match="min_coverage"):
        evaluation.authenticate_confirmation(identity, historical)
