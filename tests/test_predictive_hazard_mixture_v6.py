"""CUDA numerical contracts; execute only in the parent's serialized mlq job.

Scalar arithmetic references evaluate conditional Gaussian moment mixtures, not
CPU model fallbacks, exact sparse Bayes, or reduced-horizon research evidence.
"""

import math

import pytest
import torch

from cleanrl.plasticity.predictive_hazard_mixture_v6 import HazardMixture, SegmentEvidence
from cleanrl.plasticity.predictive_segment_posterior_v5 import SegmentPosterior
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def setup_model(dimension=4, budget=2):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    return HazardMixture(dimension, 'cuda', budget=budget)


def scalar(value):
    return torch.tensor(value, device='cuda', dtype=torch.float32)


def conditional_reference(segment, features, label):
    """Enumerate null, retained and newborn moments with scalar Gaussian PDFs.

    Consume only the saved conditional beliefs, never the implementation's
    transition weights, forecast helper, evidence, or updated branch ranking.
    The sparse factorization supplies moments; Gaussian mixing is over segments.
    """
    masses = [math.exp(value) for value in segment.log_weights.cpu().tolist()]
    total = sum(masses)
    masses = [mass / total for mass in masses]
    odds = segment.filter.log_odds[:segment.budget].cpu().tolist()
    means = segment.filter.slab_mean[:segment.budget].cpu().tolist()
    variances = segment.filter.slab_variance[:segment.budget].cpu().tolist()
    noises = segment.noise.cpu().tolist()
    hazard = segment.hazard
    branches = [((1 - hazard) * masses[0] + .5 * hazard, 0., noises[0])]
    for slot in range(segment.budget):
        mean, variance = 0., noises[slot + 1]
        for x, log_odds, m, v in zip(features, odds[slot], means[slot], variances[slot]):
            if log_odds >= 0:
                probability = 1 / (1 + math.exp(-log_odds))
            else:
                ratio = math.exp(log_odds)
                probability = ratio / (1 + ratio)
            mean += x * probability * m
            variance += x * x * (probability * v + probability * (1 - probability) * m * m)
        branches.append(((1 - hazard) * masses[slot + 1], mean, variance))
    branches.append((.5 * hazard, 0., 1 + sum(x * x for x in features) / len(features)))
    density = sum(mass * math.exp(-.5 * (label - mean) ** 2 / variance) / math.sqrt(2 * math.pi * variance)
                  for mass, mean, variance in branches)
    forecast = sum(mass * mean for mass, mean, _ in branches)
    return forecast, density


@pytest.mark.parametrize('hazard', [0., .35])
@torch.no_grad()
def test_evidence_is_full_conditional_mixture_before_top_mass_truncation(hazard):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    segment = SegmentEvidence(4, 'cuda', budget=1, hazard=hazard, noise_rate=.25)
    maximum_discard = 0.
    for x, y in [([1., 0., 1., 0.], 3.), ([1., 1., 0., 0.], -4.),
                 ([0., 0., 0., 0.], 2.), ([1., 0., 1., 0.], -2.),
                 ([0., 1., 0., 1.], .3), ([1., 1., 1., 1.], 4.)]:
        prediction, density = conditional_reference(segment, x, y)
        actual = segment.update(scalar(x), scalar(y))
        torch.testing.assert_close(actual, scalar([prediction]), rtol=3e-5, atol=2e-6)
        assert float(segment.predictive_log_prob) == pytest.approx(math.log(density), rel=3e-5, abs=2e-6)
        maximum_discard = max(maximum_discard, float(segment.discarded_mass))
    if hazard:
        assert maximum_discard > .01
    else:
        assert maximum_discard == 0.


@torch.no_grad()
def test_hyperposterior_is_sequential_bayes_and_forecast_uses_previous_weights():
    model = setup_model(budget=1)
    weights = [.25] * 4
    stream = [([1., 0., 1., 0.], 3.), ([1., 0., 1., 0.], 3.),
              ([1., 0., 1., 0.], -5.), ([0., 1., 0., 1.], 2.),
              ([0., 0., 0., 0.], -.3), ([1., 1., 0., 0.], -4.),
              ([1., 0., 1., 0.], -3.), ([1., 1., 1., 1.], 1.)]
    for x, y in stream:
        conditional = [conditional_reference(child, x, y) for child in model.segments]
        expected = [sum(weight * prediction for weight, (prediction, _) in zip(weights, conditional)),
                    *[prediction for prediction, _ in conditional]]
        forecast = model.update(scalar(x), scalar(y))
        torch.testing.assert_close(forecast, scalar(expected), rtol=5e-5, atol=2e-6)
        evidence = [weight * density for weight, (_, density) in zip(weights, conditional)]
        total = sum(evidence)
        weights = [value / total for value in evidence]
        torch.testing.assert_close(model.log_hazard_weights.exp(), scalar(weights), rtol=5e-5, atol=3e-7)
        for child, (_, density) in zip(model.segments, conditional):
            assert float(child.predictive_log_prob) == pytest.approx(math.log(density), rel=3e-5, abs=2e-6)
        assert float(model.log_hazard_weights.exp().sum()) == pytest.approx(1., abs=5e-7)
    assert max(weights) - min(weights) > 1e-4


@torch.no_grad()
def test_current_label_independence_pure_coefficients_and_complete_state_restore():
    model = setup_model()
    x = scalar([1., .5, 0., -1.])
    for y in (2., -.4, 3., 0.):
        model.update(x, scalar(y))
    saved = [tensor.clone() for tensor in model.state_tensors()]
    expected = model.mean_weights() @ x
    model.mean_weights()
    model.diagnostics()
    for tensor, original in zip(model.state_tensors(), saved):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    low = model.update(x, scalar(-8.)).clone()
    low_next = model.mean_weights().clone()
    low_state = [tensor.clone() for tensor in model.state_tensors()]
    for tensor, original in zip(model.state_tensors(), saved):
        tensor.copy_(original)
    high = model.update(x, scalar(8.)).clone()
    torch.testing.assert_close(low, high, rtol=0, atol=0)
    torch.testing.assert_close(low, expected, rtol=3e-5, atol=2e-6)
    assert not torch.allclose(low_next, model.mean_weights())
    for tensor, original in zip(model.state_tensors(), saved):
        tensor.copy_(original)
    torch.testing.assert_close(model.update(x, scalar(-8.)), low, rtol=0, atol=0)
    for tensor, original in zip(model.state_tensors(), low_state):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize('hazard', [0., 1e-4])
@torch.no_grad()
def test_child_forecasts_scales_and_retained_histories_match_actual_frozen_v5(hazard):
    model = setup_model(budget=2)
    child = model.segments[model.hazards.index(hazard)]
    frozen = SegmentPosterior(4, 'cuda', budget=2, hazard=hazard)
    for x, y in [([1., 0., 1., 0.], 4.), ([0., 1., 0., 1.], -2.),
                 ([0., 0., 0., 0.], 8.), ([1., 0., 1., 0.], -7.),
                 ([1., 1., 1., 1.], 2.), ([1., 0., 1., 0.], 3.),
                 ([0., 0., 0., 0.], .1), ([1., 1., 0., 0.], -4.)]:
        actual = model.update(scalar(x), scalar(y))[1 + model.hazards.index(hazard)]
        expected = frozen.update(scalar(x), scalar(y))[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(child.mean_weights(), frozen.mean_weights(), rtol=0, atol=0)
        torch.testing.assert_close(child.trace(), frozen.trace(), rtol=0, atol=0)
        torch.testing.assert_close(child.aggregation_weights(), frozen.aggregation_weights(), rtol=0, atol=0)
        torch.testing.assert_close(child.discarded_mass, frozen.discarded_mass, rtol=0, atol=0)


@torch.no_grad()
def test_empty_features_and_zero_hazard_keep_valid_probability_and_positive_noise():
    model = setup_model(budget=9)
    x = scalar([0., 0., 0., 0.])
    for y in (3., -2., 0., .1, 1.):
        torch.testing.assert_close(model.update(x, scalar(y)), scalar([0.] * 5), rtol=0, atol=0)
        assert torch.isfinite(model.log_hazard_weights).all()
        assert (model.log_hazard_weights.exp() > 0).all()
        assert float(model.log_hazard_weights.exp().sum()) == pytest.approx(1., abs=5e-7)
        for child in model.segments:
            density = child.predictive_log_prob.exp()
            assert torch.isfinite(density) and density > 0
            assert torch.isfinite(child.noise).all() and (child.noise > 0).all()
            posterior = child.log_weights.exp()
            prior = child.aggregation_weights()
            assert torch.isfinite(posterior).all() and (posterior >= 0).all()
            assert torch.isfinite(prior).all() and (prior >= 0).all()
            assert float(posterior.sum()) == pytest.approx(1., abs=5e-7)
            assert float(prior.sum()) == pytest.approx(1., abs=5e-7)
        assert float(model.segments[0].discarded_mass) == 0.
    torch.testing.assert_close(model.mean_weights(), torch.zeros((5, 4), device='cuda'), rtol=0, atol=0)
    assert float(model.segments[0].log_weights[0].exp()) == pytest.approx(.5, abs=3e-7)


@torch.no_grad()
def test_compiled_cuda_graph_capture_and_full_state_replay_restore():
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
    from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import Runner

    actual, reference = setup_model(), setup_model()
    cfg = sparse.Args(steps=17, input_dim=4, graph_steps=4)
    xs = torch.tensor([[True, i % 2 == 0, i % 3 == 0, i % 5 == 0] for i in range(17)], device='cuda')
    ys = scalar([3., 2., 1., -.3, -2., -5., 4., .1, 2., -1., 3., -.5, 2., 4., -3., .2, 1.])
    runner = Runner(xs, ys, actual, sparse.LinearLearner('adam', (1e-5,), cfg, xs, ys))
    control = Runner(xs, ys, reference, sparse.LinearLearner('adam', (1e-5,), cfg, xs, ys))
    initial = [tensor.clone() for tensor in runner.mutable]
    graphs = runner.capture(4)
    for tensor, original in zip(runner.mutable, initial):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    compiled = torch.compile(control.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
    for _ in range(4):
        graphs[4].replay()
    graphs[1].replay()
    for _ in range(17):
        compiled()
    torch.cuda.synchronize()
    for tensor, expected in zip(runner.mutable, control.mutable):
        torch.testing.assert_close(tensor, expected, rtol=3e-5, atol=2e-6)
    completed = [tensor.clone() for tensor in runner.mutable]
    for tensor, original in zip(runner.mutable, initial):
        tensor.copy_(original)
    for _ in range(4):
        graphs[4].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    for tensor, expected in zip(runner.mutable, completed):
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
