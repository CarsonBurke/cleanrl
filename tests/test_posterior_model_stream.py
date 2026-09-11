"""Observable posterior-predictive causality and complete graph replay contracts."""
import math

import pytest
import torch

from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
from cleanrl.plasticity import posterior_model_stream_v1 as research
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def setup_models():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    return research.ModelMixture(3, 5, 'cuda', hazard=.02)


@torch.no_grad()
def test_aggregation_uses_integrated_predictive_likelihood_after_forecast():
    model = setup_models()
    x = torch.tensor([1., -.5, .25], device='cuda')
    phi = torch.tensor([.5, -.25, .125, .8, -.4], device='cuda')
    for y in (.7, -.2, .9):
        model.update(x, phi, torch.tensor(y, device='cuda'))
    weights = model.aggregation_weights().clone()
    means, variances = [torch.zeros(1, device='cuda')], [model.noise[:1].clone()]
    for filt, data, section in zip(model.filters, (x, x / math.sqrt(3), phi), model.slices):
        mean, variance = filt.predict(data, model.noise[section])
        means.append(mean)
        variances.append(variance)
    means, variances = torch.cat(means), torch.cat(variances)
    y = torch.tensor(-.4, device='cuda')
    expected = weights @ means
    evidence = torch.distributions.Normal(means, variances.sqrt()).log_prob(y)
    expected_posterior = torch.softmax(weights[:2].log() + evidence, -1)
    prediction = model.update(x, phi, y)
    torch.testing.assert_close(prediction, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.log_weights.exp(), expected_posterior, rtol=3e-6, atol=2e-7)


@torch.no_grad()
def test_current_label_changes_learning_but_not_current_predictions():
    left, right = setup_models(), setup_models()
    x = torch.tensor([.7, -.9, .4], device='cuda')
    phi = torch.tensor([.4, -.5, .2, -.3, .6], device='cuda')
    for y in (.5, .8, -.2):
        for model in (left, right):
            model.update(x, phi, torch.tensor(y, device='cuda'))
    a = left.update(x, phi, torch.tensor(-4., device='cuda'))
    b = right.update(x, phi, torch.tensor(4., device='cuda'))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    future_a = left.frozen_predictions(x[None], phi[None])
    future_b = right.frozen_predictions(x[None], phi[None])
    assert not torch.allclose(future_a, future_b)


@torch.no_grad()
def test_switch_prior_preserves_reachable_hypotheses_without_static_probability_roundtrip():
    model = setup_models()
    model.log_weights.fill_(-1000)
    model.log_weights[:, 0] = 0
    weights = model.aggregation_weights()
    expected = model.hazard * model.log_prior.exp()
    expected[0] += 1 - model.hazard
    torch.testing.assert_close(weights[0], expected, rtol=2e-6, atol=1e-8)
    assert weights[1, 1:].eq(0).all()
    assert torch.isfinite(model.log_weights).all()


@torch.no_grad()
def test_stream_capture_restores_posterior_and_consumes_exact_tail():
    gen = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(11, 3, generator=gen, device='cuda')
    features = torch.randn(11, 5, generator=gen, device='cuda')
    ys = torch.randn(11, generator=gen, device='cuda')
    args = sparse.Args(input_dim=3, steps=11, graph_steps=4)
    def make():
        return research.StreamRunner(xs, features, ys, setup_models(),
                                     sparse.LinearLearner('adam', (.001, .003), args, xs, ys))
    actual, reference = make(), make()
    initial = [t.clone() for t in actual.mutable]
    graphs = actual.capture(4)
    for tensor, expected in zip(actual.mutable, initial):
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    for _ in range(2):
        graphs[4].replay()
    for _ in range(3):
        graphs[1].replay()
    for _ in range(11):
        reference.update()
    torch.cuda.synchronize()
    assert int(actual.index) == int(actual.baseline.index) == 11
    torch.testing.assert_close(actual.predictions, reference.predictions, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(actual.baseline_predictions, reference.baseline_predictions, rtol=3e-4, atol=3e-6)
