"""Same-target temporal contracts and unchanged paired CUDA recurrences.

CUDA tests must be submitted through mlq; no stock benchmark is run here.
The evaluator integration test owns full compiled capture restoration and the
all-68-output prelabel perturbation check.
"""

import numpy as np
import pytest
import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import stock_stream
from cleanrl.plasticity.predictive_available_state_v8 import (
    AvailableStateComparison,
    PairedAdamBank,
    build_paired_stream,
)
from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG
from cleanrl.plasticity.predictive_stock_transfer_eval_v1 import AdamBank
from cleanrl.shared import runtime


def synthetic_bars(count):
    index = np.arange(count, dtype=np.float64)
    bars = np.zeros(count, dtype=stock_stream.BAR_DTYPE)
    close = 100 * np.exp(.0001 * index + .001 * np.sin(.37 * index))
    bars['t'] = np.arange(count) * 300
    bars['open'] = close * (1 + .0002 * np.cos(.23 * index))
    bars['close'] = close
    bars['high'] = close + .1 + .03 * np.cos(.31 * index)
    bars['low'] = close - .09 - .02 * np.sin(.29 * index)
    bars['vwap'] = close + .025 * np.sin(.41 * index)
    bars['volume'] = 1000 + 170 * np.sin(.19 * index) + index
    bars['count'] = (80 + 13 * np.cos(.17 * index) + index % 11).astype(np.uint32)
    return bars


def assert_bitwise_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('lags', [1, 4, 32])
@pytest.mark.parametrize('vol_span', [.01, .07])
def test_original_frame_and_targets_are_frozen_bitwise_with_one_bar_shift(lags, vol_span):
    bars = synthetic_bars(257)
    args = stock_stream.Args(lags=lags, vol_span=vol_span)
    frozen_x, frozen_y = stock_stream.build_stream(bars, args)
    original, latest, target, _ = build_paired_stream(bars, args)
    assert_bitwise_equal(original, frozen_x)
    assert_bitwise_equal(target, frozen_y)
    assert original.shape == latest.shape == (len(bars) - lags - 1, lags * 7)
    assert target.shape == (len(bars) - lags - 1,)
    # Equal channels at equal physical bars: exactly one row forward, not a
    # new target task or a reversed lag axis. The final row is checked below.
    assert_bitwise_equal(latest[:-1], original[1:])
    assert_bitwise_equal(latest[:, :-7], original[:, 7:])


@pytest.mark.parametrize('lags', [1, 4, 32])
def test_target_bar_changes_label_but_neither_available_feature_frame(lags):
    bars = synthetic_bars(137)
    args = stock_stream.Args(lags=lags)
    original, latest, target, raw_zero = build_paired_stream(bars, args)
    row = 61
    changed = bars.copy()
    changed['close'][row + lags + 1] *= 1.0002
    changed_original, changed_latest, changed_target, changed_raw_zero = build_paired_stream(changed, args)
    assert_bitwise_equal(changed_original[:row + 1], original[:row + 1])
    assert_bitwise_equal(changed_latest[:row + 1], latest[:row + 1])
    assert changed_target[row] != target[row]
    assert_bitwise_equal(changed_target[:row], target[:row])
    assert_bitwise_equal(changed_raw_zero[:row + 1], raw_zero[:row + 1])


@pytest.mark.parametrize('lags', [1, 4, 32])
def test_intervening_available_bar_only_enters_latest_frame_without_target_shift(lags):
    bars = synthetic_bars(137)
    args = stock_stream.Args(lags=lags)
    original, latest, target, raw_zero = build_paired_stream(bars, args)
    row = 61
    changed = bars.copy()
    # Wick-only perturbation changes features without changing any returns,
    # so a target shift or new normalization cannot hide behind a label change.
    changed['high'][row + lags] += .013
    changed_original, changed_latest, changed_target, changed_raw_zero = build_paired_stream(changed, args)
    assert_bitwise_equal(changed_original[:row + 1], original[:row + 1])
    assert_bitwise_equal(changed_latest[:row], latest[:row])
    assert_bitwise_equal(changed_latest[row, :-7], latest[row, :-7])
    assert changed_latest[row, -6] != latest[row, -6]
    assert_bitwise_equal(changed_target, target)
    assert_bitwise_equal(changed_raw_zero, raw_zero)


@pytest.mark.parametrize('lags,count', [(1, 3), (4, 6), (32, 34), (32, 257)])
def test_final_real_window_and_final_real_target_are_retained(lags, count):
    bars = synthetic_bars(count)
    args = stock_stream.Args(lags=lags)
    original, latest, target, raw_zero = build_paired_stream(bars, args)
    assert len(target) == count - lags - 1
    assert_bitwise_equal(latest[-1, :-7], original[-1, 7:])
    # Independent final return-channel calculation distinguishes B-2 from
    # both the stale B-3 bar and the future B-1 target, including a one-row file.
    close = bars['close'].astype(np.float64)
    returns = np.zeros_like(close)
    returns[1:] = np.log(close[1:] / close[:-1])
    centered = returns - np.concatenate([[0.], stock_stream._ewma(returns, args.vol_span)[:-1]])
    trailing = stock_stream._ewma(np.abs(centered), args.vol_span)
    final_return_feature = np.float32(np.clip(centered[-2] / max(trailing[-3], 1e-12), -10., 10.))
    final_target = np.float32(np.clip(np.float32(centered[-1] / max(trailing[-2], 1e-12)), -10., 10.))
    assert latest[-1, -7].tobytes() == final_return_feature.tobytes()
    assert target[-1].tobytes() == final_target.tobytes()
    final_raw_zero = np.float32(np.clip(
        -stock_stream._ewma(returns, args.vol_span)[-2] / max(trailing[-2], 1e-12), -10., 10.))
    assert raw_zero[-1].tobytes() == final_raw_zero.tobytes()
    changed = bars.copy()
    changed['high'][-2] += .013
    changed_original, changed_latest, changed_target, changed_raw_zero = build_paired_stream(changed, args)
    assert_bitwise_equal(changed_original, original)
    assert_bitwise_equal(changed_latest[:-1], latest[:-1])
    assert changed_latest[-1, -6] != latest[-1, -6]
    assert_bitwise_equal(changed_target, target)
    assert_bitwise_equal(changed_raw_zero, raw_zero)


@pytest.mark.parametrize('options', [
    {'center': False}, {'raw_target': True}, {'vol_feature': True}, {'steps': 1},
    {'lags': 0}, {'lags': -1}, {'lags': True}, {'lags': 2.5},
])
def test_unsupported_stream_switches_are_rejected(options):
    with pytest.raises(ValueError):
        build_paired_stream(synthetic_bars(64), stock_stream.Args(**options))


def test_not_enough_real_bars_cannot_produce_a_fabricated_sample():
    with pytest.raises(ValueError, match='not enough bars'):
        build_paired_stream(synthetic_bars(33), stock_stream.Args())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; run via mlq')
@torch.no_grad()
def test_paired_forecasts_and_complete_states_match_independent_frozen_learners():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(9, 28, device='cuda', generator=generator)
    ys = torch.randn(9, device='cuda', generator=generator)
    model = AvailableStateComparison(28, 'cuda')
    bank = PairedAdamBank(xs, ys, stock.Args.adam_lrs)
    independent_models = [CorrelatedNIG(14, 'cuda'), CorrelatedNIG(14, 'cuda')]
    independent_banks = [AdamBank(xs[:, :14], ys, stock.Args.adam_lrs),
                         AdamBank(xs[:, 14:], ys, stock.Args.adam_lrs)]
    assert model.output_names == AvailableStateComparison.output_names
    assert len(model.output_names) + len(bank.output_names) == 68
    states = [*model.state_tensors(), *bank.mutable]
    initial = [tensor.clone() for tensor in states]
    retained = []
    for step, (x, y) in enumerate(zip(xs, ys)):
        expected = torch.cat((independent_models[0].update(x[:14], y),
                              independent_models[1].update(x[14:], y)))
        prediction = model.update(x, y)
        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
        retained.append((prediction, prediction.clone()))
        for independent in independent_banks:
            independent.update()
        bank.update()
        torch.testing.assert_close(bank.prediction,
                                   torch.cat(tuple(item.prediction for item in independent_banks)),
                                   rtol=0, atol=0)
        for paired, independent in zip((model.original, model.latest), independent_models, strict=True):
            for actual, reference in zip(paired.state_tensors(), independent.state_tensors(), strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for paired, independent in zip((bank.original, bank.latest), independent_banks, strict=True):
            for actual, reference in zip(paired.mutable, independent.mutable, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert int(model.observations) == step + 1
    # Owned forecasts cannot be overwritten by subsequent learner updates.
    for prediction, saved in retained:
        torch.testing.assert_close(prediction, saved, rtol=0, atol=0)
    torch.testing.assert_close(bank.scale, torch.cat(tuple(item.scale for item in independent_banks)),
                               rtol=0, atol=0)
    # Restore via the public lists, then replay: omitted moments, evidence,
    # or clocks make the replay diverge from the independent trajectories.
    final = [tensor.clone() for tensor in states]
    for tensor, saved in zip(states, initial, strict=True):
        tensor.copy_(saved)
    for x, y in zip(xs, ys):
        model.update(x, y)
        bank.update()
    for tensor, saved in zip(states, final, strict=True):
        torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
    for paired, independent in zip((bank.original, bank.latest), independent_banks, strict=True):
        torch.testing.assert_close(paired.mlp.predictions, independent.mlp.predictions, rtol=0, atol=0)
    diagnostics = model.diagnostics()
    for frame, independent in zip(('original', 'latest'), independent_models, strict=True):
        assert diagnostics[frame]['mixture_components'][-1] == 'null'
        np.testing.assert_allclose(diagnostics[frame]['mixture_probabilities'],
                                   independent.log_weights.exp().cpu().numpy(), rtol=0, atol=0)
