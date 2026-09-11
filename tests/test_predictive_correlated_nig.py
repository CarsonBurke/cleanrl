"""Small deterministic CUDA algebra contracts; no stock or training benchmark."""

import pytest
import torch

from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG, student_t_log_prob
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def exact_matmul_runtime():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)


def batch_posterior(xs, ys, prior):
    """Independent FP64 CUDA normal equations, not the sequential recurrence."""
    xs, ys = xs.double(), ys.double()
    identity = torch.eye(xs.shape[1], device=xs.device, dtype=torch.float64)
    precision = identity / prior + xs.T @ xs
    covariance = torch.linalg.solve(precision, identity)
    mean = torch.linalg.solve(precision, xs.T @ ys)
    # Equivalent to 1 + (y'y - m' precision m)/2, without cancellation.
    beta = 1 + .5 * ((ys - xs @ mean).square().sum() + mean.square().sum() / prior)
    return covariance, mean, beta


@torch.no_grad()
def test_dense_matches_batch_posterior_and_proper_student_t_evidence():
    priors = (1e-7, .2, 1.3)
    model = CorrelatedNIG(3, 'cuda', priors)
    xs = torch.tensor([[1., .8, -.2], [0., .7, .3], [1., 0., 0.],
                       [-.4, -.6, .2], [.7, .6, .1], [0., 0., 0.]], device='cuda')
    ys = torch.tensor([.7, -1.2, 2., -.4, 1.3, -.8], device='cuda')
    expected_log_weights = model.log_weights.clone()
    for step, (x, y) in enumerate(zip(xs, ys)):
        means, scores = [], []
        alpha = torch.tensor(2 + step / 2, device='cuda', dtype=torch.float64)
        for index, prior in enumerate(priors):
            covariance, mean, beta = batch_posterior(xs[:step], ys[:step], prior)
            torch.testing.assert_close(model.dense_cov[index].double(), covariance, rtol=3e-6, atol=3e-8)
            torch.testing.assert_close(model.dense_mean[index].double(), mean, rtol=3e-6, atol=2e-7)
            torch.testing.assert_close(model.beta[index], beta, rtol=3e-7, atol=2e-8)
            mu = x.double() @ mean
            leverage = 1 + x.double() @ covariance @ x.double()
            distribution = torch.distributions.StudentT(2 * alpha, mu, (beta * leverage / alpha).sqrt())
            score = distribution.log_prob(y.double())
            # Verify absolute proper density, not only odds where a missing
            # Student-t normalizer would cancel across experts.
            actual_score = student_t_log_prob(alpha, beta, (y.double() - mu).square(), leverage)
            torch.testing.assert_close(actual_score, score, rtol=2e-13, atol=2e-13)
            means.append(mu)
            scores.append(score)
        null_beta = 1 + .5 * ys[:step].double().square().sum()
        null = torch.distributions.StudentT(2 * alpha, torch.zeros_like(alpha), (null_beta / alpha).sqrt())
        scores.append(null.log_prob(y.double()))
        means = torch.stack(means)
        expected_mixture = (expected_log_weights[:-1].exp() * means).sum()
        prediction = model.update(x, y)
        torch.testing.assert_close(prediction[:len(priors)].double(), means, rtol=3e-6, atol=2e-7)
        torch.testing.assert_close(prediction[-2].double(), expected_mixture, rtol=3e-6, atol=2e-7)
        torch.testing.assert_close(prediction[-1], torch.zeros_like(y), rtol=0, atol=0)
        expected_log_weights += torch.stack(scores)
        expected_log_weights -= expected_log_weights.logsumexp(-1)
        torch.testing.assert_close(model.log_weights, expected_log_weights, rtol=2e-7, atol=3e-7)
        torch.testing.assert_close(model.beta[-1], 1 + .5 * ys[:step + 1].double().square().sum())
    # Include the final conditioning, not just states used by next predictions.
    for index, prior in enumerate(priors):
        covariance, mean, beta = batch_posterior(xs, ys, prior)
        torch.testing.assert_close(model.dense_cov[index].double(), covariance, rtol=3e-6, atol=3e-8)
        torch.testing.assert_close(model.dense_mean[index].double(), mean, rtol=3e-6, atol=2e-7)
        torch.testing.assert_close(model.beta[index], beta, rtol=3e-7, atol=2e-8)
    torch.testing.assert_close(model.alpha, torch.tensor(2 + len(xs) / 2, device='cuda', dtype=torch.float64))


@torch.no_grad()
def test_weak_prior_retains_leverage_evidence_below_float32_spacing_at_one():
    prior = 1e-7
    model = CorrelatedNIG(1, 'cuda', (prior,))
    x = torch.tensor([.1], device='cuda')
    model.update(x, torch.zeros((), device='cuda'))
    # y=0 cancels the residual term. Equal prior mass leaves exactly the
    # leverage penalty in dense/null posterior log odds, even below FP32 eps.
    expected = -.5 * torch.log1p(prior * x.double().square().sum())
    torch.testing.assert_close(model.log_weights[0] - model.log_weights[1], expected, rtol=2e-6, atol=3e-15)
    assert model.log_weights[0] < model.log_weights[1]


@torch.no_grad()
def test_diagonal_is_condition_then_project_not_independent_coordinate_updates():
    prior = .7
    model = CorrelatedNIG(3, 'cuda', (prior,))
    covariance = torch.eye(3, device='cuda', dtype=torch.float64) * prior
    mean = torch.zeros(3, device='cuda', dtype=torch.float64)
    beta = torch.ones((), device='cuda', dtype=torch.float64)
    xs = torch.tensor([[1., 2., -.5], [.4, 0., .8], [-.2, .7, 1.]], device='cuda')
    ys = torch.tensor([1., -.5, .7], device='cuda')
    for x, y in zip(xs, ys):
        mu = mean @ x.double()
        u = covariance @ x.double()
        leverage = 1 + x.double() @ u
        residual = y.double() - mu
        prediction = model.update(x, y)
        torch.testing.assert_close(prediction[1].double(), mu, rtol=2e-6, atol=2e-7)
        conditioned = covariance - torch.outer(u, u) / leverage
        covariance = torch.diag(conditioned.diagonal())
        mean += u * residual / leverage
        beta += residual.square() / (2 * leverage)
        torch.testing.assert_close(model.diag_cov[0].double(), covariance.diagonal(), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.diag_mean[0].double(), mean, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.beta[1], beta, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize('input_dim', [1, 3])
@torch.no_grad()
def test_one_dimensional_and_axis_aligned_orthogonal_controls_agree(input_dim):
    model = CorrelatedNIG(input_dim, 'cuda', (.03, .8))
    # Coordinate-axis observations never create off-diagonals. General rotated
    # orthogonal rows would not be an equivalent diagonal-control contract.
    xs = torch.eye(input_dim, device='cuda').repeat(2, 1)
    ys = torch.linspace(-.8, 1.2, len(xs), device='cuda')
    for x, y in zip(xs, ys):
        prediction = model.update(x, y)
        torch.testing.assert_close(prediction[:2], prediction[2:4], rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.dense_mean, model.diag_mean, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.dense_cov.diagonal(dim1=-2, dim2=-1), model.diag_cov,
                                   rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.beta[:2], model.beta[2:4], rtol=2e-7, atol=2e-8)


@torch.no_grad()
def test_correlated_information_updates_coefficient_with_zero_local_feature():
    model = CorrelatedNIG(2, 'cuda', (1.,))
    model.update(torch.tensor([1., 1.], device='cuda'), torch.tensor(0., device='cuda'))
    model.update(torch.tensor([1., 0.], device='cuda'), torch.tensor(1., device='cuda'))
    # First observation learns w1+w2=0; the second then credits w1 positively
    # and w2 negatively although its local feature is zero. Projection cannot.
    prediction = model.update(torch.tensor([0., 1.], device='cuda'), torch.tensor(0., device='cuda'))
    torch.testing.assert_close(prediction[0], torch.tensor(-.2, device='cuda'), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(prediction[1], torch.zeros((), device='cuda'), rtol=0, atol=0)


@torch.no_grad()
def test_same_compiled_callable_restored_state_is_causal_and_outputs_are_owned():
    model = CorrelatedNIG(2, 'cuda', (.1, 1.))
    update = torch.compile(model.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
    x = torch.tensor([1., .5], device='cuda')
    for value in (.7, -.2, 1.3):
        update(x, torch.tensor(value, device='cuda'))
    saved = [tensor.clone() for tensor in model.state_tensors()]

    def restore():
        for tensor, initial in zip(model.state_tensors(), saved):
            tensor.copy_(initial)

    low = update(x, torch.tensor(-4., device='cuda'))
    low_copy = low.clone()
    low_future = update(x, torch.tensor(0., device='cuda')).clone()
    restore()
    high = update(x, torch.tensor(4., device='cuda'))
    torch.testing.assert_close(low, high, rtol=0, atol=0)
    high_future = update(x, torch.tensor(0., device='cuda')).clone()
    assert not torch.allclose(low_future[:-1], high_future[:-1])
    torch.testing.assert_close(low, low_copy, rtol=0, atol=0)
    # Restore and replay on the SAME compiled callable, covering every mutable
    # posterior statistic (including mixture probabilities and scale clocks).
    restore()
    replay = update(x, torch.tensor(-4., device='cuda'))
    torch.testing.assert_close(replay, low_copy, rtol=0, atol=0)
    torch.testing.assert_close(update(x, torch.tensor(0., device='cuda')), low_future, rtol=0, atol=0)
    assert int(model.observations.cpu()) == 5


@torch.no_grad()
def test_zero_inputs_leave_all_experts_equal_to_calibrated_unknown_variance_null():
    model = CorrelatedNIG(3, 'cuda', (.01, 1.))
    x = torch.zeros(3, device='cuda')
    ys = torch.tensor([0., 1.5, -2., .25, 0., -.7], device='cuda')
    initial_log_weights = model.log_weights.clone()
    initial_covariance = model.dense_cov.clone()
    for step, y in enumerate(ys, start=1):
        prediction = model.update(x, y)
        torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)
        beta = 1 + .5 * ys[:step].double().square().sum()
        expected_noise = beta / (1 + step / 2)
        torch.testing.assert_close(model.noise, expected_noise.expand_as(model.noise), rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(model.log_weights, initial_log_weights, rtol=0, atol=2e-14)
    torch.testing.assert_close(model.dense_cov, initial_covariance, rtol=0, atol=0)
