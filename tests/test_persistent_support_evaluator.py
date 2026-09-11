"""Censored/phase-causal reporting and genuine independently trained v3 controls."""
import numpy as np
import pytest
import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
from cleanrl.plasticity.predictive_persistent_support_eval_v4 import ComparisonModel, summarize_comparison
from cleanrl.shared import runtime


NAMES = ('v4_likelihood_birth_only', 'v4_likelihood_switching',
         'v3_likelihood_static', 'v3_likelihood_switching', 'adam_0.001',
         'single_likelihood', 'single_cohort')


def observations():
    clean = np.arange(12, dtype=np.float64) / 7
    target = clean + np.array([.2, -.5, 1., -.3, .7, -.1, .6, -.4, .8, -.6, .9, -.7])
    prediction = np.arange(12 * len(NAMES), dtype=np.float64).reshape(12, len(NAMES)) / 11
    endpoints = (1, 3, 6, 7, 10, 12)
    risks = (9., 8., 7., 2., 5., 4.)
    rows, start = [], 0
    for end, risk in zip(endpoints, risks):
        rows.append({'interval_start': start, 'step': end,
                     'prequential': stock.metric_sums(target[start:end], prediction[start:end]),
                     'prequential_clean_mse': np.square(prediction[start:end] - clean[start:end, None]).mean(0).tolist(),
                     'exact_risk': {'clean_mse': [risk + i for i in range(len(NAMES))]}})
        start = end
    return rows, clean, target, prediction


def test_comparison_weights_actual_observations_and_pairs_both_objectives():
    rows, clean, target, prediction = observations()
    summary = summarize_comparison(rows, NAMES, 'change', 12, 'adam_0.001')
    full = summary['clean_metrics']['full']
    expected = np.square(prediction - clean[:, None]).mean(0)
    np.testing.assert_allclose(full['clean_mse'], expected)
    assert full['count'] == 12 and full['complete'] and summary['full_horizon']
    # Unequal intervals must not receive equal vote in the full-stream metric.
    assert not np.allclose(expected, np.mean([row['prequential_clean_mse'] for row in rows], axis=0))
    for phase, bounds in {'prefix': (0, 3), 'suffix_before_change': (3, 6),
                          'suffix_after_change': (6, 12), 'suffix': (3, 12)}.items():
        start, end = bounds
        expected_phase = np.square(prediction[start:end] - clean[start:end, None]).mean(0)
        np.testing.assert_allclose(summary['clean_metrics'][phase]['clean_mse'], expected_phase)
        reference = stock.metric_sums(target[start:end], prediction[start:end])
        np.testing.assert_allclose([m['error_ratio'] for m in summary['phase_metrics'][phase]],
                                   [m['error_ratio'] for m in reference])
    assert summary['primary_minus_control_clean_mse']['v3_likelihood_static']['full'] == pytest.approx(expected[0] - expected[2])
    assert summary['primary_minus_control_clean_mse']['single_likelihood']['full'] == pytest.approx(expected[0] - expected[5])
    assert summary['secondary_switching_minus_v3_switching_clean_mse']['full'] == pytest.approx(expected[1] - expected[3])
    assert summary['primary_minus_control_clean_mse']['adam_0.001']['full'] == pytest.approx(expected[0] - expected[4])
    relapse = summary['postchange_relapse']
    assert relapse['checkpoint_steps'] == [6, 7, 10, 12]
    assert relapse['final_minus_best_clean_mse'] == [2.] * len(NAMES)
    assert relapse['maximum_rise_from_running_best'] == [3.] * len(NAMES)


def test_switch_checkpoint_changes_risk_not_consumed_interval_phase():
    rows, clean, _, prediction = observations()
    summary = summarize_comparison(rows[:3], NAMES, 'change', 12)
    assert not summary['full_horizon']
    assert summary['processed_observations'] == 6
    assert summary['clean_metrics']['prefix']['complete']
    assert not summary['clean_metrics']['full']['complete']
    changed = summary['clean_metrics']['suffix_after_change']
    assert changed['count'] == 0 and changed['clean_mse'] is None and not changed['complete']
    assert summary['phase_metrics']['suffix_after_change'] is None
    np.testing.assert_allclose(summary['clean_metrics']['full']['clean_mse'],
                               np.square(prediction[:6] - clean[:6, None]).mean(0))
    assert summary['postchange_relapse']['checkpoint_steps'] == [6]
    assert summary['final_risk']['clean_mse'] == [7. + i for i in range(len(NAMES))]


def test_summary_rejects_unsplittable_phase_and_gapped_evidence():
    rows, _, _, _ = observations()
    with pytest.raises(ValueError, match='split exactly'):
        summarize_comparison([{**rows[0], 'step': 4}], NAMES, 'change', 12)
    with pytest.raises(ValueError, match='contiguous'):
        summarize_comparison([rows[0], rows[2]], NAMES, 'change', 12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_compiled_comparison_keeps_real_v3_trajectory_and_prelabel_causality():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    paired = ComparisonModel(5, 'cuda', periods=(4,))
    frozen = PredictiveMeanRisk(5, 'cuda')
    updates = [torch.compile(model.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
               for model in (paired, frozen)]
    xs = torch.tensor([[1., 0., .2, 0., -.1], [0., 1., 0., .3, 0.],
                       [1., 0., .1, 0., 0.], [.3, 1., 0., 0., -.2],
                       [1., .4, 0., .2, 0.], [0., 1., -.1, 0., .1],
                       [1., 0., .2, 0., -.1]], device='cuda')
    ys = torch.tensor([.8, -.3, 1.2, .5, -.6, 1., .7], device='cuda')
    indices = torch.tensor((0, 1, 10, 11), device='cuda')
    label = torch.empty((), device='cuda')
    for step, (x, y) in enumerate(zip(xs, ys)):
        label.copy_(y)
        before = [value.clone() for value in paired.state_tensors()]
        expected = paired.mean_weights() @ x
        actual = updates[0](x, label).clone()
        control = updates[1](x, y).index_select(0, indices)
        torch.testing.assert_close(actual[-4:], control, rtol=3e-5, atol=3e-7)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)
        torch.testing.assert_close(paired.mean_weights()[-4:], frozen.mean_weights().index_select(0, indices),
                                   rtol=3e-5, atol=3e-7)
        torch.testing.assert_close(actual[paired.output_names.index('single_cohort')],
                                   actual[paired.output_names.index('v4_cohort_0')],
                                   rtol=3e-5, atol=3e-7)
    # Replay the SAME compiled callable from identical state and input storage.
    # Separate instances can compile to different FP32 reduction/fusion orders.
    learned = paired.mean_weights().clone()
    for value, saved in zip(paired.state_tensors(), before):
        value.copy_(saved)
    label.add_(5)
    other = updates[0](x, label)
    torch.testing.assert_close(actual, other, rtol=0, atol=0)
    assert not torch.allclose(paired.mean_weights(), learned)
