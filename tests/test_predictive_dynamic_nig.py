"""Tiny deterministic CUDA algebra/capture contracts, not stock benchmarks."""

import math

import pytest
import torch

from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG, student_t_log_prob
from cleanrl.plasticity.predictive_dynamic_nig_v6 import DynamicNIG
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def exact_matmul_runtime():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)


def observations():
    xs = torch.tensor([[1., .8, -.2], [0., .7, .3], [1., 0., 0.],
                       [-.4, -.6, .2], [.7, .6, .1], [0., 0., 0.]], device='cuda')
    ys = torch.tensor([.7, -1.2, 2., -.4, 1.3, -.8], device='cuda')
    return xs, ys


def weighted_normal_equations(xs, ys, prior, halflife):
    """Independent FP64 batch solve, including the faded initial precision."""
    xs, ys = xs.double(), ys.double()
    count, dim = xs.shape
    delta = 1. if halflife == 0 else 2 ** (-1 / halflife)
    weights = delta ** torch.arange(count - 1, -1, -1, device=xs.device, dtype=torch.float64)
    identity = torch.eye(dim, device=xs.device, dtype=torch.float64)
    precision = delta ** count / prior * identity + xs.T @ (weights[:, None] * xs)
    covariance = torch.linalg.solve(precision, identity)
    mean = torch.linalg.solve(precision, xs.T @ (weights * ys))
    return covariance, mean


@torch.no_grad()
def test_static_experts_and_control_equal_frozen_dense_v5():
    priors = (1e-7, .2, 1.3)
    model = DynamicNIG(3, 'cuda', priors, (2.5, 11))
    frozen = CorrelatedNIG(3, 'cuda', priors)
    xs, ys = observations()
    k = len(priors)
    for x, y in zip(xs, ys):
        actual, expected = model.update(x, y), frozen.update(x, y)
        torch.testing.assert_close(actual[4:4 + k], expected[:k], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[-2], rtol=0, atol=0)
        torch.testing.assert_close(model.mean[:k], frozen.dense_mean, rtol=0, atol=0)
        torch.testing.assert_close(model.cov[:k], frozen.dense_cov, rtol=0, atol=0)
        torch.testing.assert_close(model.beta[:k], frozen.beta[:k], rtol=0, atol=0)
        torch.testing.assert_close(model.beta[-1], frozen.beta[-1], rtol=0, atol=0)
        torch.testing.assert_close(model.alpha, frozen.alpha, rtol=0, atol=0)
        torch.testing.assert_close(model.log_static, frozen.log_weights, rtol=0, atol=0)


@torch.no_grad()
def test_dynamic_mean_and_covariance_match_weighted_normal_equations():
    model = DynamicNIG(3, 'cuda', (.2, 1.3), (2.5, 11))
    xs, ys = observations()
    for step, (x, y) in enumerate(zip(xs, ys)):
        expected_forecasts = []
        for cfg in model.configs:
            _, mean = weighted_normal_equations(xs[:step], ys[:step], cfg['prior'], cfg['halflife'])
            expected_forecasts.append(x.double() @ mean)
        prediction = model.update(x, y)
        torch.testing.assert_close(prediction[4:].double(), torch.stack(expected_forecasts),
                                   rtol=5e-6, atol=3e-7)
        for index, cfg in enumerate(model.configs):
            covariance, mean = weighted_normal_equations(xs[:step + 1], ys[:step + 1],
                                                          cfg['prior'], cfg['halflife'])
            torch.testing.assert_close(model.cov[index].double(), covariance, rtol=5e-6, atol=3e-7)
            torch.testing.assert_close(model.mean[index].double(), mean, rtol=5e-6, atol=3e-7)


@torch.no_grad()
def test_independent_fp64_dlm_evidence_and_three_causal_mixtures():
    priors, halflives = (1e-7, .7), (2.5, 11)
    model = DynamicNIG(3, 'cuda', priors, halflives)
    xs, ys = observations()
    k, nstatic = len(model.configs), len(priors)
    covariances = torch.stack([torch.eye(3, device='cuda', dtype=torch.float64) * cfg['prior']
                              for cfg in model.configs])
    means = torch.zeros((k, 3), device='cuda', dtype=torch.float64)
    betas = torch.ones(k + 1, device='cuda', dtype=torch.float64)
    alpha = torch.tensor(2., device='cuda', dtype=torch.float64)
    deltas = torch.tensor([1. if cfg['halflife'] == 0 else 2 ** (-1 / cfg['halflife'])
                           for cfg in model.configs], device='cuda', dtype=torch.float64)
    # Build independent prior masses rather than copying the learner's weights.
    log_all = torch.tensor([*([.25 / nstatic] * nstatic),
                            *([.25 / (k - nstatic)] * (k - nstatic)), .5],
                           device='cuda', dtype=torch.float64).log()
    log_static = torch.tensor([*([.5 / nstatic] * nstatic), .5],
                              device='cuda', dtype=torch.float64).log()
    log_dynamic = torch.tensor([*([.5 / (k - nstatic)] * (k - nstatic)), .5],
                               device='cuda', dtype=torch.float64).log()
    for x32, y32 in zip(xs, ys):
        x, y = x32.double(), y32.double()
        forecast = torch.cat((means @ x, torch.zeros(1, device='cuda', dtype=torch.float64)))
        expected_mixtures = torch.stack(((log_all.exp() * forecast).sum(),
                                         (log_static[:-1].exp() * forecast[:nstatic]).sum(),
                                         (log_dynamic[:-1].exp() * forecast[nstatic:-1]).sum()))
        covariances = covariances / deltas[:, None, None]
        u = covariances @ x
        leverage = torch.cat((1 + (u * x).sum(-1), torch.ones_like(alpha).reshape(1)))
        residual = y - forecast
        distribution = torch.distributions.StudentT(2 * alpha, forecast,
                                                      (betas * leverage / alpha).sqrt())
        scores = distribution.log_prob(y)
        # Absolute normalized density, not just relative odds.
        torch.testing.assert_close(student_t_log_prob(alpha, betas, residual.square(), leverage),
                                   scores, rtol=2e-13, atol=2e-13)
        prediction = model.update(x32, y32)
        torch.testing.assert_close(prediction[:3].double(), expected_mixtures, rtol=5e-6, atol=3e-7)
        torch.testing.assert_close(prediction[4:].double(), forecast[:-1], rtol=5e-6, atol=3e-7)
        torch.testing.assert_close(prediction[3], torch.zeros_like(y32), rtol=0, atol=0)
        log_all = log_all + scores
        log_static = log_static + torch.cat((scores[:nstatic], scores[-1:]))
        log_dynamic = log_dynamic + scores[nstatic:]
        for log_weights in (log_all, log_static, log_dynamic):
            log_weights.sub_(log_weights.logsumexp(-1))
        torch.testing.assert_close(model.log_all, log_all, rtol=3e-7, atol=4e-7)
        torch.testing.assert_close(model.log_static, log_static, rtol=3e-7, atol=4e-7)
        torch.testing.assert_close(model.log_dynamic, log_dynamic, rtol=3e-7, atol=4e-7)
        covariances = covariances - u[:, :, None] * u[:, None, :] / leverage[:-1, None, None]
        means = means + u * (residual[:-1] / leverage[:-1])[:, None]
        betas = betas + .5 * residual.square() / leverage
        alpha = alpha + .5
        torch.testing.assert_close(model.beta, betas, rtol=3e-7, atol=4e-7)
        torch.testing.assert_close(model.noise, betas / (alpha - 1), rtol=3e-7, atol=4e-7)
        for state in model.state_tensors():
            assert torch.isfinite(state).all()
        assert (torch.linalg.eigvalsh(model.cov.double()) > 0).all()


@torch.no_grad()
def test_zero_input_inflates_geometry_without_decaying_or_resetting_mean():
    model = DynamicNIG(2, 'cuda', (.7,), (2.5,))
    model.update(torch.tensor([1., .5], device='cuda'), torch.tensor(2., device='cuda'))
    mean, covariance, beta = model.mean.clone(), model.cov.clone(), model.beta.clone()
    assert torch.count_nonzero(mean) == mean.numel()
    zero = torch.zeros(2, device='cuda')
    prediction = model.update(zero, torch.tensor(-3., device='cuda'))
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)
    torch.testing.assert_close(model.mean, mean, rtol=0, atol=0)
    torch.testing.assert_close(model.cov[0], covariance[0], rtol=0, atol=0)
    torch.testing.assert_close(model.cov[1].double(), covariance[1].double() * 2 ** (1 / 2.5),
                               rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(model.beta, beta + 4.5, rtol=0, atol=0)


@torch.no_grad()
def test_forecasts_are_prelabel_owned_and_restoring_state_restores_future():
    model = DynamicNIG(2, 'cuda', (.1, 1.), (2.5,))
    x = torch.tensor([1., .5], device='cuda')
    for value in (.7, -.2, 1.3):
        model.update(x, torch.tensor(value, device='cuda'))
    saved = [tensor.clone() for tensor in model.state_tensors()]

    def restore():
        for tensor, initial in zip(model.state_tensors(), saved):
            tensor.copy_(initial)

    low = model.update(x, torch.tensor(-4., device='cuda'))
    low_copy = low.clone()
    low_weights = [tensor.clone() for tensor in (model.log_all, model.log_static, model.log_dynamic)]
    low_future = model.update(x, torch.tensor(0., device='cuda')).clone()
    restore()
    high = model.update(x, torch.tensor(4., device='cuda'))
    torch.testing.assert_close(low, high, rtol=0, atol=0)
    for actual, previous in zip((model.log_all, model.log_static, model.log_dynamic), low_weights):
        assert not torch.allclose(actual, previous)
    high_future = model.update(x, torch.tensor(0., device='cuda'))
    assert not torch.allclose(low_future[:3], high_future[:3])
    torch.testing.assert_close(low, low_copy, rtol=0, atol=0)
    restore()
    torch.testing.assert_close(model.update(x, torch.tensor(-4., device='cuda')), low_copy, rtol=0, atol=0)
    torch.testing.assert_close(model.update(x, torch.tensor(0., device='cuda')), low_future, rtol=0, atol=0)
    assert int(model.observations.cpu()) == 5


@torch.no_grad()
def test_complete_state_restores_same_compiled_cuda_graph():
    model = DynamicNIG(2, 'cuda', (.1, 1.), (2.5,))
    x = torch.tensor([1., .5], device='cuda')
    y = torch.tensor(.7, device='cuda')
    model.update(x, y)
    # These eight public statistics are the capture/checkpoint contract.
    public_state = [model.mean, model.cov, model.alpha, model.beta, model.log_all,
                    model.log_static, model.log_dynamic, model.observations]
    initial = [tensor.clone() for tensor in public_state]

    def restore():
        for tensor, saved in zip(model.state_tensors(), initial, strict=True):
            tensor.copy_(saved)

    compiled = torch.compile(model.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            compiled(x, y)
            restore()
            compiled(x, y)
        stream.synchronize()
        restore()
        expected = compiled(x, y).clone()
        expected_state = [tensor.clone() for tensor in public_state]
        restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(x, y)
        restore()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        for actual, reference in zip(public_state, expected_state):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        # Restore and replay the SAME graph with another label. Forecast stays
        # identical; all three posterior weight vectors must react to that label.
        restore()
        y.fill_(-4.)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        for actual, reference in zip(public_state[4:7], expected_state[4:7]):
            assert not torch.allclose(actual, reference)
    finally:
        stream.synchronize()
        restore()
        torch.cuda.synchronize()


@pytest.mark.parametrize('kwargs', [
    {'priors': ()}, {'priors': (0.,)}, {'priors': (math.inf,)},
    {'priors': (math.nan,)}, {'priors': (.1, .1)},
    {'halflives': ()}, {'halflives': (-1.,)}, {'halflives': (math.inf,)},
    {'halflives': (math.nan,)}, {'halflives': (2., 2.)},
])
def test_invalid_prior_or_halflife_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DynamicNIG(2, 'cuda', **kwargs)
