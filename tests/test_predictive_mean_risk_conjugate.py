"""Proper residual-scale priors, causal scoring, and complete CUDA capture state."""

import math

import pytest
import torch

from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def setup_model(hazard=1e-4, noise_rate=.001):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    return PredictiveMeanRisk(5, 'cuda', hazard=hazard, noise_rate=noise_rate)


def compiled_update(model):
    return torch.compile(model.update, fullgraph=True, mode='max-autotune-no-cudagraphs')


def warmup(*updates):
    x = torch.tensor([1., -.5, .25, .7, -.3], device='cuda')
    for value in (.7, -.2, 1.3, .5, .9):
        y = torch.tensor(value, device='cuda')
        for update in updates:
            update(x, y)
    return x


@torch.no_grad()
def test_zero_stream_remains_finite_after_200_half_discount_updates():
    model = setup_model(noise_rate=.5)
    update = compiled_update(model)
    x, y = torch.zeros(5, device='cuda'), torch.zeros((), device='cuda')
    forecasts = torch.stack([update(x, y) for _ in range(200)])
    torch.testing.assert_close(forecasts, torch.zeros_like(forecasts), rtol=0, atol=0)
    for state in model.state_tensors():
        assert torch.isfinite(state).all()
    assert (model.noise > 0).all() and model.energy > 0
    # The retained prior contributes 2 / (2 + sum_{j=0}^{199} .5**j),
    # rather than the vanishing .5**200 of the old residual EMA.
    expected = 2 / (2 + sum(.5 ** j for j in range(200)))
    torch.testing.assert_close(model.noise, torch.full_like(model.noise, expected))
    torch.testing.assert_close(model.energy, torch.full_like(model.energy, expected))
    torch.testing.assert_close(model.aggregation_weights(), model.log_prior.exp().expand(5, -1))


@pytest.mark.parametrize('noise_rate', [0., .5])
@torch.no_grad()
def test_scoring_uses_previous_conjugate_predictive_variance(noise_rate):
    model = setup_model(hazard=.2, noise_rate=noise_rate)
    update = compiled_update(model)
    xs = torch.tensor([[1., -.5, .25, .7, -.3], [0., .4, -.6, .1, .8],
                       [1., 0., 0., 0., 0.], [-.3, .8, .2, -.7, .4],
                       [.5, .3, -.1, .8, -.2], [-.2, .1, .6, -.3, .7]], device='cuda')
    ys = torch.tensor([.7, -1.2, 2., -.4, 1.3, -.8], device='cuda')
    residuals = []
    expected_noise = torch.ones(9, device='cuda', dtype=torch.float64)
    expected_energy = torch.ones((), device='cuda', dtype=torch.float64)
    for step, (x, y) in enumerate(zip(xs, ys)):
        # The inverse-Gamma mean is also the zero-mean Student-t predictive
        # variance; the expert adds its coefficient uncertainty for scoring.
        torch.testing.assert_close(model.noise, expected_noise.float())
        torch.testing.assert_close(model.energy, expected_energy.float())
        sparse_mean, sparse_variance = model.filter.predict(x, expected_noise[1:].float())
        means = torch.cat((torch.zeros(1, device='cuda'), sparse_mean))
        variances = torch.cat((expected_noise[:1].float(), sparse_variance))
        prior_log = model.aggregation_weights()[:4].log()
        likelihood = -.5 * (math.log(2 * math.pi) + variances.log() + (y - means).square() / variances)
        risk = (2 * y * means - means.square()) / (2 * expected_energy.float())
        posterior = prior_log + torch.stack((likelihood, likelihood, risk, risk))
        expected_weights = posterior - posterior.logsumexp(-1, keepdim=True)
        prediction = update(x, y)
        torch.testing.assert_close(prediction[5:], means, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(model.log_weights, expected_weights, rtol=3e-5, atol=2e-6)

        # Build an independent closed-form oracle from ACTUAL pre-label
        # residuals, not post-update coefficients or the model's statistics.
        residuals.append((y.double() - prediction[5:].double()).square())
        powers = torch.tensor([(1 - noise_rate) ** (step - j) for j in range(step + 1)],
                              device='cuda', dtype=torch.float64)
        count = powers.sum()
        residual_sum = (powers[:, None] * torch.stack(residuals)).sum(0)
        target_sum = (powers * ys[:step + 1].double().square()).sum()
        expected_noise = (1 + residual_sum / 2) / (2 + count / 2 - 1)
        expected_energy = (1 + target_sum / 2) / (2 + count / 2 - 1)
        torch.testing.assert_close(model.noise, expected_noise.float())
        torch.testing.assert_close(model.energy, expected_energy.float())


@pytest.mark.parametrize('hazard', [0., 1e-4, 1.])
@torch.no_grad()
def test_current_label_changes_learning_not_current_forecasts(hazard):
    left, right = setup_model(hazard, .5), setup_model(hazard, .5)
    update_left, update_right = compiled_update(left), compiled_update(right)
    x = warmup(update_left, update_right)
    a = update_left(x, torch.tensor(-4., device='cuda'))
    b = update_right(x, torch.tensor(4., device='cuda'))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    future_a, future_b = x @ left.mean_weights().T, x @ right.mean_weights().T
    assert not torch.allclose(future_a, future_b)
    assert torch.isfinite(future_a).all() and torch.isfinite(future_b).all()


@torch.no_grad()
def test_effective_coefficients_predict_all_prelabel_outputs():
    model = setup_model(noise_rate=.5)
    update = compiled_update(model)
    warmup(update)
    xs = torch.tensor([[0., .4, -.6, .1, .8], [1., 0., 0., 0., 0.],
                       [-.3, .8, .2, -.7, .4]], device='cuda')
    ys = torch.tensor([-.7, 1.5, .2], device='cuda')
    for x, y in zip(xs, ys):
        expected = x @ model.mean_weights().T
        actual = update(x, y)
        assert actual.shape == (14,)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)


@torch.no_grad()
def test_compiled_runner_restores_capture_and_consumes_nondivisible_tail():
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
    from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import Runner

    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    gen = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(11, 5, device='cuda', generator=gen)
    ys = torch.randn(11, device='cuda', generator=gen)
    args = sparse.Args(input_dim=5, steps=11, graph_steps=4)

    def make():
        return Runner(xs, ys, setup_model(noise_rate=.5),
                      sparse.LinearLearner('adam', (.001, .003), args, xs, ys))

    actual, reference = make(), make()
    graphs = actual.capture(4)
    for _ in range(2):
        graphs[4].replay()
    for _ in range(3):
        graphs[1].replay()
    reference_update = compiled_update(reference)
    for _ in range(11):
        reference_update()
    torch.cuda.synchronize()
    assert int(actual.index) == int(actual.adam.index) == int(actual.adam.steps) == 11
    torch.testing.assert_close(actual.predictions, reference.predictions, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(actual.model.mean_weights(), reference.model.mean_weights(), rtol=3e-4, atol=3e-6)
