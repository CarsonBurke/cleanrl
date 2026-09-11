"""CUDA mathematical/graph contracts; run through mlq, not as training evidence."""

import math

import pytest
import torch

from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def tensor(value):
    return torch.tensor(value, device="cuda", dtype=torch.float32)


def log_density(residual, variance):
    return -0.5 * (math.log(2 * math.pi) + variance.log() + residual.square() / variance)


@torch.no_grad()
def test_one_dimensional_static_posterior_matches_integrated_conjugate_bayes():
    configs = [{"prior": 1.7, "inclusion": .18, "hazard": 0.0},
               {"prior": .4, "inclusion": .65, "hazard": 0.0}]
    model = SparsePosterior(1, configs, "cuda")
    m = torch.zeros(2, device="cuda", dtype=torch.float64)
    v = torch.tensor([1.7, .4], device="cuda", dtype=torch.float64)
    odds = torch.tensor([math.log(.18 / .82), math.log(.65 / .35)], device="cuda", dtype=torch.float64)
    # Later nonzero slab means distinguish conditional-active from mixed residuals.
    for feature, label, noise in [(1.2, 2.1, .7), (-.8, .3, 1.1), (.5, -.4, .2)]:
        x, y, r = tensor([feature]), tensor(label), tensor(noise)
        p = odds.sigmoid()
        expected_mean = p * m * feature
        expected_variance = noise + feature**2 * (p * v + p * (1 - p) * m.square())
        prediction, uncertainty = model.update(x, y, r)
        torch.testing.assert_close(prediction.double(), expected_mean, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(uncertainty.double(), expected_variance, rtol=2e-6, atol=2e-7)
        on_variance = noise + feature**2 * v
        residual_on = label - feature * m
        odds += log_density(residual_on, on_variance) - log_density(
            torch.full_like(m, label), torch.full_like(v, noise))
        m = m + v * feature * residual_on / on_variance
        v = v * noise / on_variance
        torch.testing.assert_close(model.log_odds[:, 0].double(), odds, rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(model.slab_mean[:, 0].double(), m, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.slab_variance[:, 0].double(), v, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(model.mean_weights()[:, 0].double(), odds.sigmoid() * m, rtol=3e-6, atol=3e-7)


@torch.no_grad()
def test_uncertain_known_active_other_coordinate_enters_both_likelihoods():
    model = SparsePosterior(2, [{"prior": 1.0, "inclusion": .3, "hazard": 0.0}], "cuda")
    model.log_odds.copy_(tensor([[math.log(.3 / .7), math.inf]]))
    model.slab_mean.copy_(tensor([[-.4, .8]]))
    model.slab_variance.copy_(tensor([[.6, 1.5]]))
    x, y, r = tensor([1.3, -.7]), tensor(.9), tensor(.4)
    # Integrating the independent known-active second coefficient is exact on
    # this update; treating its mean as certain gives a different Bayes factor.
    x0, x1 = x.double()
    m0, m1 = model.slab_mean[0].double()
    v0, v1 = model.slab_variance[0].double()
    off_residual = y.double() - x1 * m1
    off_variance = r.double() + x1.square() * v1
    on_variance = off_variance + x0.square() * v0
    expected_odds = model.log_odds[0, 0].double() + log_density(off_residual - x0 * m0, on_variance) - log_density(
        off_residual, off_variance)
    expected_mean = m0 + v0 * x0 * (off_residual - x0 * m0) / on_variance
    expected_variance = v0 * off_variance / on_variance
    model.update(x, y, r)
    torch.testing.assert_close(model.log_odds[0, 0].double(), expected_odds, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.slab_mean[0, 0].double(), expected_mean, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.slab_variance[0, 0].double(), expected_variance, rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_zero_residual_is_evidence_against_unnecessarily_uncertain_slab():
    model = SparsePosterior(1, [{"prior": 3.0, "inclusion": .5, "hazard": 0.0}], "cuda")
    model.update(tensor([2.0]), tensor(0.0), tensor(.5))
    expected_odds = -.5 * math.log1p(4 * 3 / .5)
    torch.testing.assert_close(model.log_odds, tensor([[expected_odds]]))
    assert model.log_odds.sigmoid().item() < .5
    torch.testing.assert_close(model.slab_mean, tensor([[0.0]]), rtol=0, atol=0)


@torch.no_grad()
def test_zero_features_carry_no_observation_information_even_when_others_are_active():
    model = SparsePosterior(3, [{"prior": 1.0, "inclusion": .2, "hazard": 0.0}], "cuda")
    model.slab_mean.copy_(tensor([[.4, -.7, 1.3]]))
    model.slab_variance.copy_(tensor([[.3, .8, .6]]))
    before = [s.clone() for s in model.state_tensors()]
    forecast = model.update(tensor([0., 0., 0.]), tensor(37.0), tensor(.4))
    torch.testing.assert_close(forecast[0], tensor([0.0]), rtol=0, atol=0)
    torch.testing.assert_close(forecast[1], tensor([.4]), rtol=0, atol=0)
    for actual, expected in zip(model.state_tensors(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.update(tensor([0., 1., 0.]), tensor(-5.0), tensor(.4))
    for actual, expected in zip(model.state_tensors(), before):
        torch.testing.assert_close(actual[:, [0, 2]], expected[:, [0, 2]], rtol=0, atol=0)
    assert model.slab_mean[0, 1] != before[1][0, 1]


@torch.no_grad()
def test_forecast_is_label_independent_nonmutating_and_uses_next_prior():
    configs = [{"prior": 1.2, "inclusion": .4, "hazard": .3}]
    left, right = [SparsePosterior(2, configs, "cuda") for _ in range(2)]
    for model in (left, right):
        model.update(tensor([1., -.5]), tensor(1.8), tensor(.7))
    before = [s.clone() for s in left.state_tensors()]
    x, r = tensor([.8, 1.1]), tensor(.6)
    predicted = left.predict(x, r)
    torch.testing.assert_close(predicted[0], (left.mean_weights() * x).sum(-1))
    for actual, expected in zip(left.state_tensors(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual in (left.update(x, tensor(-4.), r), right.update(x, tensor(7.), r)):
        for value, expected in zip(actual, predicted):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not torch.allclose(left.mean_weights(), right.mean_weights())


@torch.no_grad()
def test_dense_one_dimensional_filter_matches_gaussian_posterior_sequentially():
    model = SparsePosterior(1, [{"prior": 2.0, "inclusion": 1.0, "hazard": 0.0}], "cuda")
    precision, information = 1 / 2., 0.
    for feature, label, noise in [(1.2, .9, .4), (-.3, 1.1, .8), (.7, -.2, .6)]:
        prediction, uncertainty = model.update(tensor([feature]), tensor(label), tensor(noise))
        torch.testing.assert_close(prediction, tensor([feature * information / precision]))
        torch.testing.assert_close(uncertainty, tensor([noise + feature**2 / precision]))
        precision += feature**2 / noise
        information += feature * label / noise
        torch.testing.assert_close(model.mean_weights(), tensor([[information / precision]]))
        torch.testing.assert_close(model.slab_variance, tensor([[1 / precision]]))
        torch.testing.assert_close(model.log_odds.sigmoid(), tensor([[1.0]]), rtol=0, atol=0)
        assert torch.isposinf(model.log_odds).all()


@torch.no_grad()
def test_dense_first_update_matches_multivariate_gaussian_marginals():
    model = SparsePosterior(2, [{"prior": 1.3, "inclusion": 1.0, "hazard": 0.0}], "cuda")
    x, y, r = tensor([.7, -1.2]), tensor(1.6), tensor(.8)
    covariance = 1.3 * torch.eye(2, device="cuda", dtype=torch.float64)
    xd = x.double()
    denominator = r.double() + xd @ covariance @ xd
    gain = covariance @ xd / denominator
    posterior_covariance = covariance - gain[:, None] * (xd @ covariance)[None, :]
    model.update(x, y, r)
    torch.testing.assert_close(model.mean_weights()[0].double(), gain * y.double(), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.slab_variance[0].double(), posterior_covariance.diag(), rtol=2e-6, atol=2e-7)
    # The off-diagonal covariance is deliberately not represented: this is
    # a first-update marginal check, not an exact joint-Bayes claim.


@torch.no_grad()
def test_reset_mixture_matches_raw_moments_and_zero_feature_applies_only_transition():
    configs = [{"prior": 1.7, "inclusion": .25, "hazard": h} for h in (0., .3, 1.)]
    configs += [{"prior": .9, "inclusion": 1., "hazard": .4}]
    model = SparsePosterior(2, configs, "cuda")
    model.log_odds.copy_(tensor([[.8, -1.1]] * 3 + [[math.inf, math.inf]]))
    model.slab_mean.copy_(tensor([[.7, -1.2]] * 4))
    model.slab_variance.copy_(tensor([[.4, .8]] * 4))
    p, m, v = model.log_odds.double().sigmoid(), model.slab_mean.double(), model.slab_variance.double()
    h = torch.tensor([[c["hazard"]] for c in configs], device="cuda", dtype=torch.float64)
    p0 = torch.tensor([[c["inclusion"]] for c in configs], device="cuda", dtype=torch.float64)
    v0 = torch.tensor([[c["prior"]] for c in configs], device="cuda", dtype=torch.float64)
    pp = (1 - h) * p + h * p0
    next_mean = (1 - h) * p * m
    next_second_moment = (1 - h) * p * (v + m.square()) + h * p0 * v0
    next_variance = next_second_moment - next_mean.square()
    x, r = tensor([.6, -1.1]), tensor(.7)
    forecast, uncertainty = model.predict(x, r)
    torch.testing.assert_close(forecast.double(), (next_mean * x.double()).sum(-1), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(uncertainty.double(), r.double() + (next_variance * x.double().square()).sum(-1), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.mean_weights().double(), next_mean, rtol=2e-6, atol=2e-7)
    a = (1 - h) * p / pp
    slab_mean = a * m
    slab_variance = a * v + (1 - a) * v0 + a * (1 - a) * m.square()
    model.update(tensor([0., 0.]), tensor(30.), r)
    torch.testing.assert_close(model.log_odds.sigmoid().double(), pp, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.slab_mean.double(), slab_mean, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.slab_variance.double(), slab_variance, rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_dense_reset_matches_total_variance_law_without_inclusion_drift():
    model = SparsePosterior(1, [{"prior": 1.6, "inclusion": 1., "hazard": .25}], "cuda")
    model.slab_mean.fill_(1.2)
    model.slab_variance.fill_(.3)
    next_mean = .75 * 1.2
    next_variance = .75 * .3 + .25 * 1.6 + .75 * .25 * 1.2**2
    prediction, uncertainty = model.update(tensor([1.]), tensor(-.4), tensor(.6))
    torch.testing.assert_close(prediction, tensor([next_mean]))
    torch.testing.assert_close(uncertainty, tensor([.6 + next_variance]))
    torch.testing.assert_close(model.slab_mean, tensor([[next_mean + next_variance * (-.4 - next_mean) / (.6 + next_variance)]]))
    torch.testing.assert_close(model.slab_variance, tensor([[next_variance * .6 / (.6 + next_variance)]]))
    assert torch.isposinf(model.log_odds).all()


@pytest.mark.parametrize("mixed_hazards", [False, True])
@torch.no_grad()
def test_static_finite_log_odds_can_recover_after_sigmoid_saturates(mixed_hazards):
    configs = [{"prior": 1., "inclusion": .5, "hazard": 0.}]
    if mixed_hazards:
        configs += [{"prior": 1., "inclusion": .5, "hazard": .2}]
    model = SparsePosterior(1, configs, "cuda")
    model.log_odds.fill_(25.)
    model.slab_mean.fill_(5.)
    model.slab_variance.fill_(.2)
    assert model.log_odds[0, 0].sigmoid() == 1
    # The static row must preserve its finite log odds through a transition
    # even when another row needs a reset. A sigmoid/logit round-trip loses it.
    model.update(tensor([0.]), tensor(0.), tensor(.2))
    torch.testing.assert_close(model.log_odds[0, 0], tensor(25.), rtol=0, atol=0)
    expected = 25. - .5 * (25. / .4 + math.log(2.))
    model.update(tensor([1.]), tensor(0.), tensor(.2))
    torch.testing.assert_close(model.log_odds[0, 0], tensor(expected), rtol=2e-6, atol=2e-6)
    assert model.log_odds[0, 0].sigmoid() < .01


@torch.no_grad()
def test_per_expert_observation_variances_match_independent_filters():
    configs = [{"prior": 1.2, "inclusion": .3, "hazard": 0.},
               {"prior": .7, "inclusion": .6, "hazard": .2},
               {"prior": .9, "inclusion": 1., "hazard": .1}]
    batched = SparsePosterior(2, configs, "cuda")
    independent = [SparsePosterior(2, [config], "cuda") for config in configs]
    # K != D catches accidentally broadcasting expert noise along coordinates.
    for features, label, noises in [([1., -.4], .8, [.2, .8, 1.3]),
                                     ([.7, 1.], -1.3, [1.1, .4, .6])]:
        x, y, r = tensor(features), tensor(label), tensor(noises)
        predicted = batched.predict(x, r)
        returned = batched.update(x, y, r)
        for actual, expected in zip(returned, predicted):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for k, model in enumerate(independent):
            expected_forecast = model.update(x, y, r[k])
            for actual, expected in zip(returned, expected_forecast):
                torch.testing.assert_close(actual[k:k + 1], expected, rtol=2e-6, atol=2e-7)
            for actual, expected in zip(batched.state_tensors(), model.state_tensors()):
                torch.testing.assert_close(actual[k:k + 1], expected, rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_compiled_cuda_replay_matches_eager_and_restores_every_mutable_state():
    configs = [{"prior": 1.2, "inclusion": .3, "hazard": 0.},
               {"prior": .7, "inclusion": .6, "hazard": .2},
               {"prior": .9, "inclusion": 1., "hazard": .1}]
    actual, eager = [SparsePosterior(3, configs, "cuda") for _ in range(2)]
    x, y, r = tensor([1., -.4, 0.]), tensor(.8), tensor(.6)
    initial = [s.clone() for s in actual.state_tensors()]
    state_pointers = [s.data_ptr() for s in actual.state_tensors()]
    compiled = torch.compile(actual.update, fullgraph=True, options={"triton.cudagraphs": False})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(x, y, r)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(x, y, r)
    for current, original in zip(actual.state_tensors(), initial):
        current.copy_(original)
    examples = [([1., -.4, 0.], .8, .6), ([0., .7, 1.], -1.3, .9), ([.3, 0., -.8], 2.1, .4)]
    for features, label, noise in examples:
        x.copy_(tensor(features))
        y.copy_(tensor(label))
        r.copy_(tensor(noise))
        expected_forecast = eager.update(x, y, r)
        graph.replay()
        for value, expected in zip(captured, expected_forecast):
            torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6)
        for value, expected in zip(actual.state_tensors(), eager.state_tensors()):
            torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6)
    final = [s.clone() for s in actual.state_tensors()]
    for current, original in zip(actual.state_tensors(), initial):
        current.copy_(original)
    for features, label, noise in examples:
        x.copy_(tensor(features))
        y.copy_(tensor(label))
        r.copy_(tensor(noise))
        graph.replay()
    for value, expected in zip(actual.state_tensors(), final):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert [s.data_ptr() for s in actual.state_tensors()] == state_pointers
