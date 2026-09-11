"""CUDA behavioral contracts. Execute only inside the parent's serialized mlq job.

Analytic references are scalar arithmetic, not CPU model fallbacks. These short
contracts are correctness tests, never reduced-horizon research evidence.
"""

import math

import pytest
import torch

from cleanrl.plasticity.predictive_segment_posterior_v5 import SegmentPosterior
from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def setup_model(dimension=1, budget=2, hazard=.2, noise_rate=.25):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    return SegmentPosterior(dimension, 'cuda', budget=budget, hazard=hazard, noise_rate=noise_rate)


def scalar(value):
    return torch.tensor(value, device='cuda', dtype=torch.float32)


@torch.no_grad()
def test_static_d1_known_noise_kernel_matches_conjugate_posterior():
    # D1 with a nontrivial spike probability defends the exact special case.
    kernel = SparsePosterior(1, [dict(prior=1., inclusion=.3, hazard=0.)], 'cuda')
    odds, mean, variance = math.log(.3 / .7), 0., 1.
    observation_variance = 1.7
    for x, y in [(1., 2.), (.5, -.4), (0., 8.), (-1., 1.3), (2., -.7)]:
        probability = 1 / (1 + math.exp(-odds))
        prediction = probability * mean * x
        total = observation_variance + x * x * (probability * variance + probability * (1 - probability) * mean * mean)
        actual, actual_variance = kernel.update(scalar([x]), scalar(y), scalar([observation_variance]))
        torch.testing.assert_close(actual, scalar([prediction]), rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(actual_variance, scalar([total]), rtol=2e-5, atol=2e-7)
        on_variance = observation_variance + variance * x * x
        odds += -.5 * (math.log(on_variance / observation_variance)
                        + (y - mean * x) ** 2 / on_variance - y * y / observation_variance)
        mean += variance * x * (y - mean * x) / on_variance
        variance *= observation_variance / on_variance
    expected = mean / (1 + math.exp(-odds))
    torch.testing.assert_close(kernel.mean_weights(), scalar([[expected]]), rtol=2e-5, atol=2e-7)


class ScalarSegmentReference:
    """Independent dense D1 enumeration with proper segment-local IG statistics."""

    def __init__(self, budget, hazard, rate):
        self.budget, self.hazard, self.rate = budget, hazard, rate
        self.null_mass, self.null_n, self.null_q = .5, 0., 0.
        self.branches = [dict(mass=.5, mean=0., variance=1., n=0., q=0., birth=0)]
        self.t = 0

    def coefficient(self):
        return sum((1 - self.hazard) * b['mass'] * b['mean'] for b in self.branches)

    def update(self, x, y):
        prediction = self.coefficient() * x
        branches = [dict(b, mass=b['mass'] * (1 - self.hazard)) for b in self.branches]
        if self.hazard:
            branches.append(dict(mass=.5 * self.hazard, mean=0., variance=1., n=0., q=0., birth=self.t))
        null_prior = (1 - self.hazard) * self.null_mass + .5 * self.hazard
        null_noise = (2 + self.null_q) / (2 + self.null_n)
        null_score = math.log(null_prior) - .5 * (math.log(2 * math.pi * null_noise) + y * y / null_noise)
        scores = []
        for branch in branches:
            noise = (2 + branch['q']) / (2 + branch['n'])
            residual = y - branch['mean'] * x
            total = noise + branch['variance'] * x * x
            scores.append(math.log(branch['mass']) - .5 * (math.log(2 * math.pi * total) + residual * residual / total))
            branch['mean'] += branch['variance'] * x * residual / total
            branch['variance'] *= noise / total
            branch['n'] = (1 - self.rate) * branch['n'] + 1
            branch['q'] = (1 - self.rate) * branch['q'] + residual * residual
        offset = max(null_score, *scores)
        unnormalized = [math.exp(null_score - offset), *[math.exp(s - offset) for s in scores]]
        normalizer = sum(unnormalized)
        masses = [mass / normalizer for mass in unnormalized]
        for branch, mass in zip(branches, masses[1:]):
            branch['mass'] = mass
        branches.sort(key=lambda branch: branch['mass'], reverse=True)
        discarded = sum(branch['mass'] for branch in branches[self.budget:])
        retained = masses[0] + sum(branch['mass'] for branch in branches[:self.budget])
        self.null_mass = masses[0] / retained
        self.branches = branches[:self.budget]
        for branch in self.branches:
            branch['mass'] /= retained
        self.null_n = (1 - self.rate) * self.null_n + 1
        self.null_q = (1 - self.rate) * self.null_q + y * y
        self.t += 1
        return prediction, discarded


@pytest.mark.parametrize('budget,hazard', [(1, .35), (3, .2), (3, 0.)])
@torch.no_grad()
def test_future_forecasts_preserve_gathered_segment_and_null_scale_statistics(budget, hazard):
    model = setup_model(budget=budget, hazard=hazard)
    reference = ScalarSegmentReference(budget, hazard, .25)
    # Large reversals make newborns outrank old branches; zeros update IG scale
    # without coefficient evidence. Future forecasts expose a wrong gather/reset.
    stream = [(1., 4.), (1., 4.), (0., 8.), (1., -7.), (.2, -1.),
              (1., -5.), (0., .1), (-1., 3.), (1., 0.), (1., 6.), (.3, 2.)]
    for x, y in stream:
        expected, discarded = reference.update(x, y)
        actual = model.update(scalar([x]), scalar(y))
        torch.testing.assert_close(actual, scalar([expected]), rtol=8e-5, atol=2e-6)
        torch.testing.assert_close(model.mean_weights(), scalar([[reference.coefficient()]]), rtol=8e-5, atol=2e-6)
        assert float(model.discarded_mass) == pytest.approx(discarded, rel=1e-4, abs=2e-7)
        assert float(model.log_weights.exp().sum()) == pytest.approx(1., abs=3e-7)
        assert float(model.aggregation_weights().sum()) == pytest.approx(1., abs=3e-7)
    if hazard:
        assert any(branch['birth'] > 0 for branch in reference.branches)
    else:
        assert float(model.cumulative_discarded_mass) == 0.


@torch.no_grad()
def test_current_label_independence_and_pure_next_forecast():
    left, right = setup_model(dimension=4), setup_model(dimension=4)
    x = scalar([1., .5, 0., -1.])
    for y in (2., -.4, 3., 0.):
        left.update(x, scalar(y))
        right.update(x, scalar(y))
    before = [t.clone() for t in left.state_tensors()]
    expected = left.mean_weights() @ x
    torch.testing.assert_close(expected, left.mean_weights() @ x, rtol=0, atol=0)
    for tensor, original in zip(left.state_tensors(), before):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    low = left.update(x, scalar(-20.))
    high = right.update(x, scalar(20.))
    torch.testing.assert_close(low, high, rtol=0, atol=0)
    torch.testing.assert_close(low, expected, rtol=3e-5, atol=2e-7)
    assert not torch.allclose(left.mean_weights(), right.mean_weights())


@torch.no_grad()
def test_null_zero_features_do_not_create_coefficients_or_nonfinite_zero_mass_slots():
    model = setup_model(dimension=4, budget=9, hazard=0.)
    x = scalar([0., 0., 0., 0.])
    for y in (10., -3., 0., .1):
        torch.testing.assert_close(model.update(x, scalar(y)), scalar([0.]), rtol=0, atol=0)
    torch.testing.assert_close(model.mean_weights(), torch.zeros((1, 4), device='cuda'), rtol=0, atol=0)
    assert float(model.log_weights[0].exp()) == pytest.approx(.5, abs=2e-7)
    assert torch.isfinite(model.noise).all()
    assert float(model.cumulative_discarded_mass) == 0.


@torch.no_grad()
def test_whole_state_cuda_graph_replay_restores_and_matches_compiled_reference():
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
    from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import Runner

    actual = SegmentPosterior(4, 'cuda', budget=2, hazard=.3)
    reference = SegmentPosterior(4, 'cuda', budget=2, hazard=.3)
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
    # Replaying identical saved complete state must give identical ranking/ties.
    completed = [tensor.clone() for tensor in runner.mutable]
    for tensor, original in zip(runner.mutable, initial):
        tensor.copy_(original)
    for _ in range(4):
        graphs[4].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    for tensor, expected in zip(runner.mutable, completed):
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@torch.no_grad()
def test_traced_runner_records_each_consumed_observation_through_final_row():
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
    from cleanrl.plasticity.predictive_segment_eval_v5 import AdamWBank, ComparisonModel, TracedRunner, adam_grid

    cfg = sparse.Args(steps=4, input_dim=4, graph_steps=1)
    xs = torch.tensor([[True, False, False, True], [False, True, True, False],
                       [True, True, False, False], [False, False, True, True]], device='cuda')
    ys = scalar([1., -.5, 2., .25])
    model = ComparisonModel(4, 'cuda')
    runner = TracedRunner(xs, ys, model, AdamWBank(adam_grid()[:1], cfg, xs, ys))
    graphs = runner.capture(1)
    for row in range(len(xs)):
        graphs[1].replay()
        torch.cuda.synchronize()
        for trace, segment in zip(runner.traces, model.segments):
            torch.testing.assert_close(trace[row], segment.trace(), rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(runner.discards[row],
                                   torch.stack([s.discarded_mass for s in model.segments]),
                                   rtol=0, atol=0)
    assert int(runner.index) == len(xs)
