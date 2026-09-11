"""Independent numerical contracts; Main queues CUDA execution, never CPU models."""
import math

import numpy as np
import pytest
import torch
from scipy.stats import t as student_t

from cleanrl.plasticity.predictive_structure_mixture_v7 import (
    SingletonSegment, StructuralMixture, bucket_resample,
)
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def cuda(values):
    return torch.tensor(values, device='cuda', dtype=torch.float64)


def reference(history_x, history_y, x, y):
    """Integrate Gaussian regression independently via observation-space kernel.

    For each mutually exclusive support use C=I+X X' and beta=1+y'C^-1y/2.
    This deliberately does not use the production sufficient-statistic formula.
    """
    n, dimension = history_x.shape
    logs, means, densities, scales = [], [], [], []
    alpha = 2 + n / 2
    for support in (None, *range(dimension)):
        design = np.zeros((n, 0)) if support is None else history_x[:, support:support + 1]
        kernel = np.eye(n) + design @ design.T
        beta = 1 + history_y @ np.linalg.solve(kernel, history_y) / 2
        precision = np.eye(design.shape[1]) + design.T @ design
        covariance = np.linalg.inv(precision)
        mean = covariance @ design.T @ history_y
        feature = np.zeros(0) if support is None else x[support:support + 1]
        location = feature @ mean
        scale2 = beta / alpha * (1 + feature @ covariance @ feature)
        prior = .5 if support is None else .5 / dimension
        logs.append(math.log(prior) - .5 * np.linalg.slogdet(kernel)[1] - alpha * math.log(beta))
        means.append(float(location))
        scales.append(scale2)
        densities.append(student_t.pdf(y, df=2 * alpha, loc=location, scale=math.sqrt(scale2)))
    posterior = np.exp(np.array(logs) - max(logs))
    posterior /= posterior.sum()
    return posterior, posterior @ means, math.log(posterior @ densities), np.array(scales)


@torch.no_grad()
def test_exact_singleton_posterior_and_student_t_against_batch_integration():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    model = SingletonSegment(3, 'cuda', budget=1, hazard=0., compression='top_mass')
    xs = np.array([[1., 0., -1.], [0., 2., .3], [.5, 0., 0.], [2., -1., 0.]])
    ys = np.array([1.2, -.7, 2.1, -.4])
    for i, (x, y) in enumerate(zip(xs, ys)):
        p, forecast, density, _ = reference(xs[:i], ys[:i], x, y)
        actual_p = model.posterior()[0][0].exp().cpu().numpy()
        np.testing.assert_allclose(actual_p, p, rtol=1e-12, atol=1e-12)
        actual = model.update(cuda(x), cuda(y))
        assert float(actual) == pytest.approx(forecast, rel=1e-12, abs=1e-12)
        assert float(model.predictive_log_prob) == pytest.approx(density, rel=1e-12, abs=1e-12)
    p, _, _, _ = reference(xs, ys, xs[0], ys[0])
    np.testing.assert_allclose(model.posterior()[0][0].exp().cpu().numpy(), p, rtol=1e-12, atol=1e-12)


@torch.no_grad()
def test_absent_features_retain_prior_support_odds_but_learn_noise():
    model = SingletonSegment(4, 'cuda', budget=1, hazard=0., compression='top_mass')
    for y in (3., -2., 0., .5):
        assert float(model.update(cuda([0., 0., 0., 0.]), cuda(y))) == 0.
    support, mean, precision, alpha, beta = model.posterior()
    torch.testing.assert_close(support[0].exp(), cuda([.5, .125, .125, .125, .125]), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mean[0], cuda([0., 0., 0., 0.]), rtol=0, atol=0)
    torch.testing.assert_close(beta[0], cuda([7.625] * 5), rtol=0, atol=0)
    assert float(alpha[0]) == 4.


@torch.no_grad()
def test_bucket_mass_and_hand_selected_categorical_representatives():
    # At observation10: ages 1,1,2,3,4; buckets 0,0,1,2,3.
    log_mass = cuda([.1, .2, .3, .15, .25]).log()
    birth = torch.tensor([10, 10, 9, 8, 7], device='cuda', dtype=torch.int64)
    observation = torch.tensor(10, device='cuda', dtype=torch.int64)
    for uniform, expected in ((.2, 0), (.8, 1)):
        chosen, masses = bucket_resample(log_mass, birth, observation, cuda([uniform, .4, .7, .1, .9]), 5)
        assert int(chosen[0]) == expected
        torch.testing.assert_close(masses[:4].exp(), cuda([.3, .3, .15, .25]), rtol=1e-12, atol=1e-12)
        assert torch.isneginf(masses[4])
        assert float(masses.exp().sum()) == pytest.approx(1., abs=1e-12)
    # Tiny global mass remains in log space and has a usable local distribution.
    tiny = cuda([-1001., -1000., 0.])
    chosen, masses = bucket_resample(tiny, birth[:3], observation, cuda([.9, .2, .3]), 3)
    assert int(chosen[0]) == 1
    assert float(masses[0]) == pytest.approx(-1000 + math.log1p(math.exp(-1)), abs=1e-12)
    assert torch.isfinite(masses[0])


@torch.no_grad()
def test_zero_hazard_compression_equals_static_exact_posterior():
    bucket = SingletonSegment(3, 'cuda', hazard=0.)
    exact = SingletonSegment(3, 'cuda', budget=1, hazard=0., compression='top_mass')
    for i in range(40):
        x = cuda([i % 2, i % 3 == 0, 1.])
        y = cuda((i % 5) - 2.)
        torch.testing.assert_close(bucket.update(x, y, cuda([.73] * 32)), exact.update(x, y), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bucket.predictive_log_prob, exact.predictive_log_prob, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bucket.mean_weights(), exact.mean_weights(), rtol=1e-12, atol=1e-12)
        assert float(bucket.mass_error) == pytest.approx(0., abs=1e-12)
        assert int(torch.isfinite(bucket.log_weights).sum()) == 1
        assert float(bucket.cumulative_discarded_mass) == 0.


@torch.no_grad()
def test_bucket_evidence_scores_full_birth_mixture_before_compression():
    model = SingletonSegment(2, 'cuda', hazard=.2)
    xs, ys = [], []
    for i in range(12):
        x = np.array([float(i % 2), 1.])
        y = float(i % 4 - 1)
        histories = [(int(model.birth[j]), float(model.log_weights[j].exp()) * .8)
                     for j in range(32) if torch.isfinite(model.log_weights[j])]
        histories.append((i, .2))
        means, densities = [], []
        for birth, mass in histories:
            hx = np.asarray(xs[birth:]).reshape(-1, 2)
            hy = np.asarray(ys[birth:])
            _, mean, density, _ = reference(hx, hy, x, y)
            means.append(mass * mean)
            densities.append(mass * math.exp(density))
        actual = model.update(cuda(x), cuda(y), cuda([.37] * 32))
        assert float(actual) == pytest.approx(sum(means), abs=2e-12)
        assert float(model.predictive_log_prob) == pytest.approx(math.log(sum(densities)), abs=2e-12)
        assert float(model.log_weights.exp().sum()) == pytest.approx(1., abs=2e-12)
        assert float(model.discarded_mass) == 0.
        xs.append(x)
        ys.append(y)


@torch.no_grad()
def test_top_mass_ablation_reports_exact_removed_posterior_mass():
    model = SingletonSegment(2, 'cuda', budget=1, hazard=.4, compression='top_mass')
    model.update(cuda([1., 0.]), cuda(4.))
    # Existing history and fresh prior must both be scored before the budget cut.
    _, old_mean, old_log, _ = reference(np.array([[1., 0.]]), np.array([4.]), np.array([1., 0.]), -3.)
    _, new_mean, new_log, _ = reference(np.empty((0, 2)), np.empty(0), np.array([1., 0.]), -3.)
    masses = np.array([.6 * math.exp(old_log), .4 * math.exp(new_log)])
    forecast = model.update(cuda([1., 0.]), cuda(-3.))
    assert float(forecast) == pytest.approx(.6 * old_mean + .4 * new_mean, abs=1e-12)
    assert float(model.discarded_mass) == pytest.approx(masses.min() / masses.sum(), abs=1e-12)
    assert float(model.predictive_log_prob) == pytest.approx(math.log(masses.sum()), abs=1e-12)


@torch.no_grad()
def test_structural_forecast_is_prelabel_and_updates_each_evidence_once():
    tape = cuda(np.full((8, 4, 32), .47))
    left, right = StructuralMixture(3, 'cuda', tape), StructuralMixture(3, 'cuda', tape)
    x = cuda([1., 0., 1.]).float()
    left.update(x, cuda(2.).float())
    for destination, source in zip(right.state_tensors(), left.state_tensors()):
        destination.copy_(source)
    before = [t.clone() for t in left.state_tensors()]
    prediction = left.mean_weights() @ x.double()
    structure_prior = left.log_structure_weights.clone()
    hazard_prior = left.singleton.log_hazard_weights.clone()
    left.mean_weights()
    left.diagnostics()
    for actual, saved in zip(left.state_tensors(), before):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    a, b = left.update(x, cuda(-4.).float()), right.update(x, cuda(7.).float())
    torch.testing.assert_close(a, prediction, rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(left.log_structure_weights.exp(),
                               torch.softmax(structure_prior + left.structure_log_prob, 0), rtol=1e-12, atol=1e-12)
    child_log = torch.stack([s.predictive_log_prob for s in left.singleton.segments])
    torch.testing.assert_close(left.singleton.log_hazard_weights.exp(),
                               torch.softmax(hazard_prior + child_log, 0), rtol=1e-12, atol=1e-12)
    assert not torch.allclose(left.mean_weights(), right.mean_weights())
    for actual, saved in zip(left.state_tensors(), before):
        actual.copy_(saved)
    replay = left.update(x, cuda(-4.).float())
    torch.testing.assert_close(replay, a, rtol=0, atol=0)
    assert int(left.singleton.uniforms_consumed) == 256


@torch.no_grad()
def test_nonfinite_label_remains_a_numerical_failure_not_zero_model():
    model = SingletonSegment(2, 'cuda')
    model.update(cuda([1., 0.]), cuda(float('nan')), cuda([.5] * 32))
    assert not torch.isfinite(model.predictive_log_prob)
    assert not torch.isfinite(model.mean_weights()).all()
