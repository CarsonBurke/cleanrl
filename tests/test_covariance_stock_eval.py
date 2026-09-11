"""Focused contracts; run CUDA tests through mlq, never during concurrent editing."""

import numpy as np
import pytest
import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import network_bayes_stream_v2 as bayes
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def make_pair(method="network"):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    generator = torch.Generator(device="cuda").manual_seed(1)
    args = bayes.Args(input_dim=3, hidden=2, diffusion=1e-5, graph_steps=1)
    initial = bayes.init_weights(args, generator, torch.device("cuda"))
    xs = torch.randn(12, 3, device="cuda", generator=generator)
    ys = torch.randn(12, device="cuda", generator=generator)
    grid = (1.0,) if method == "network" else (3e-5, 1e-3)
    original = bayes.Learner(method, grid, initial, args, xs, ys, ys, torch.ones_like(ys))
    measured = stock.MeasuredLearner(method, grid, initial, args, xs, ys)
    return original, measured


@torch.no_grad()
@pytest.mark.parametrize("method", ["network", "adam"])
def test_wrapper_preserves_frozen_initialization_and_every_posterior_update(method):
    original, measured = make_pair(method)
    for actual, expected in zip(measured.mutable, original.mutable):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for step in range(len(original.xs)):
        before = bayes.sample_state(original.weights, original.xs[step])[0].clone()
        original.update()
        measured.update()
        torch.testing.assert_close(measured.predictions[step], before, rtol=0, atol=0)
        for actual, expected in zip(measured.mutable, original.mutable):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_graph_records_prediction_before_label_update_and_restores_capture_consumption():
    original, measured = make_pair()
    # These labels force an observable pre/post difference in the first update.
    measured.ys[0] = 7.0
    first_prediction = bayes.sample_state(measured.weights, measured.xs[0])[0].clone()
    graph, _ = measured.capture()
    assert measured.index.item() == 0
    assert measured.steps.item() == 0
    assert torch.count_nonzero(measured.predictions).item() == 0
    for step in range(len(measured.xs)):
        before = bayes.sample_state(original.weights, original.xs[step])[0].clone()
        original.update()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(measured.predictions[step], before, rtol=3e-3, atol=3e-5)
        if step == 0:
            torch.testing.assert_close(measured.predictions[0], first_prediction, rtol=3e-3, atol=3e-5)
            after = bayes.sample_state(measured.weights, measured.xs[0])[0]
            assert (after - measured.predictions[0]).abs().max().item() > 0.1
    assert measured.index.item() == len(measured.xs)
    assert measured.steps.item() == len(measured.xs)
    for actual, expected in zip(measured.mutable, original.mutable):
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-5)


def test_metric_decomposition_separates_prediction_energy_from_signed_fit():
    target = np.array([1.0, -2.0, 3.0, -4.0])
    predictions = np.stack([target, -target, np.zeros_like(target)], axis=1)
    perfect, wrong_sign, zero = stock.metric_sums(target, predictions)
    assert perfect["error_ratio"] == 0.0
    assert wrong_sign["error_ratio"] == 4.0
    assert zero["error_ratio"] == 1.0
    assert perfect["prediction_energy_ratio"] == wrong_sign["prediction_energy_ratio"] == 1.0
    assert perfect["signed_cross_term_ratio"] == -2.0
    assert wrong_sign["signed_cross_term_ratio"] == 2.0
    for metrics in (perfect, wrong_sign, zero):
        assert metrics["error_ratio"] == pytest.approx(
            1 + metrics["prediction_energy_ratio"] + metrics["signed_cross_term_ratio"])
        assert metrics["decomposition_residual"] == pytest.approx(0)
        assert metrics["target_squared_sum"] == 30.0


def test_prefix_lock_rejects_suffix_consumption_and_uses_only_scored_prefix():
    phases = stock.phase_ranges(40, 2)
    assert phases["selection_prefix"] == (2, 10)
    assert phases["suffix_positive"] == (10, 25)
    assert phases["suffix_reversed"] == (25, 40)
    target = np.ones(40)
    predictions = np.zeros((40, 2))
    predictions[:2, 0] = 100.0  # Cold-start penalty must not choose candidate 1.
    predictions[2:10, 0] = 1.0
    predictions[10:, 1] = 1.0  # Suffix would reverse the choice if consulted.
    choice = stock.select_prefix(predictions[:10], target[:10], (1e-5, 1e-3), 2, 10, 10)
    assert choice["selected_lr"] == 1e-5
    assert choice["suffix_observations_used"] == 0
    assert choice["optimizer_updates_at_lock"] == 10
    assert choice["candidates"][0]["error_ratio"] == 0
    assert choice["candidates"][1]["error_ratio"] == 1
    with pytest.raises(ValueError, match="exactly"):
        stock.select_prefix(predictions[:10], target[:10], (1e-5, 1e-3), 2, 10, 11)
    with pytest.raises(ValueError, match="exactly"):
        stock.select_prefix(predictions, target, (1e-5, 1e-3), 2, 10, 10)


def test_views_preserve_magnitude_and_lock_planted_standardization_before_suffix():
    features = np.zeros((40, 224), dtype=np.float32)
    features[:, 217] = np.linspace(-2, 3, 40)
    target = np.linspace(-1, 1, 40, dtype=np.float32)
    views, signal, metadata = stock.build_views(features, target, 10, 25, 1)
    np.testing.assert_array_equal(np.abs(views["random_sign"]), np.abs(target))
    np.testing.assert_allclose(views["planted"] - target, signal, atol=1e-7)
    source = features[:, 217].astype(np.float64)
    z = ((source - source[:10].mean()) / source[:10].std()).astype(np.float32)
    np.testing.assert_array_equal(signal[:25], np.float32(0.03) * z[:25])
    np.testing.assert_array_equal(signal[25:], -np.float32(0.03) * z[25:])
    modified = features.copy()
    modified[10:, 217] += 1000
    _, changed_signal, changed_metadata = stock.build_views(modified, target, 10, 25, 1)
    assert metadata == changed_metadata
    np.testing.assert_array_equal(signal[:10], changed_signal[:10])
    repeat, _, _ = stock.build_views(features, target, 10, 25, 1)
    np.testing.assert_array_equal(views["random_sign"], repeat["random_sign"])


def test_paired_response_scores_added_signal_not_total_market_mse():
    phases = stock.phase_ranges(40, 2)
    real = np.full((40, 1), 20.0)
    signal = np.linspace(-0.03, 0.03, 40)
    planted = real + signal[:, None]
    response = stock.paired_response(real, planted, signal, phases)
    for rows in response["phase_metrics"].values():
        assert rows[0]["error_ratio"] == pytest.approx(0.0, abs=1e-20)
        assert rows[0]["prediction_energy_ratio"] == pytest.approx(1.0)
