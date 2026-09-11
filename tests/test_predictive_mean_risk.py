"""Causal forecasts and genuine likelihood-versus-mean-risk ablation contracts."""

import pytest
import torch

from cleanrl.plasticity.predictive_mean_risk_v2 import PredictiveMeanRisk
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def setup_model(hazard=1e-4):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    return PredictiveMeanRisk(5, 'cuda', hazard=hazard)


def warmup(*models):
    x = torch.tensor([1., -.5, .25, .7, -.3], device='cuda')
    for value in (.7, -.2, 1.3, .5, .9):
        y = torch.tensor(value, device='cuda')
        for model in models:
            model.update(x, y)
    return x


@pytest.mark.parametrize('hazard', [0., 1e-4, 1.])
@torch.no_grad()
def test_current_label_changes_learning_not_current_forecasts(hazard):
    left, right = setup_model(hazard), setup_model(hazard)
    x = warmup(left, right)
    a = left.update(x, torch.tensor(-4., device='cuda'))
    b = right.update(x, torch.tensor(4., device='cuda'))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    future_a, future_b = x @ left.mean_weights().T, x @ right.mean_weights().T
    assert not torch.allclose(future_a, future_b)
    assert torch.isfinite(future_a).all() and torch.isfinite(future_b).all()


@torch.no_grad()
def test_mean_risk_ignores_expert_variance_at_fixed_forecasts():
    left, right = setup_model(), setup_model()
    x = warmup(left, right)
    # Alter only uncertainty, not expert means, aggregation state, or common S.
    # These are real sparse learners: their subsequent updates remain distinct.
    right.noise.mul_(torch.arange(1., 10., device='cuda'))
    right.filter.slab_variance.mul_(7.)
    prior = left.aggregation_weights()[2:4]
    energy = left.energy.clone()
    y = torch.tensor(1.7, device='cuda')
    a, b = left.update(x, y), right.update(x, y)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    reward = (2 * y * a[5:] - a[5:].square()) / (2 * energy)
    expected = torch.softmax(prior.log() + reward, -1)
    torch.testing.assert_close(left.log_weights[2:4].exp(), expected, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(right.log_weights[2:4], left.log_weights[2:4], rtol=0, atol=0)
    assert not torch.allclose(left.log_weights[:2], right.log_weights[:2])


@torch.no_grad()
def test_effective_coefficients_predict_all_prelabel_outputs():
    model = setup_model()
    warmup(model)
    xs = torch.tensor([[0., .4, -.6, .1, .8], [1., 0., 0., 0., 0.],
                       [-.3, .8, .2, -.7, .4]], device='cuda')
    ys = torch.tensor([-.7, 1.5, .2], device='cuda')
    for x, y in zip(xs, ys):
        expected = x @ model.mean_weights().T
        actual = model.update(x, y)
        assert actual.shape == (14,)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)


@torch.no_grad()
def test_logspace_evidence_recovers_expert_after_probability_underflow():
    model = setup_model()
    x = torch.tensor([1., 0., 0., 0., 0.], device='cuda')
    for _ in range(8):
        model.update(x, torch.tensor(3., device='cuda'))
    # Represent overwhelming past evidence against the sparse family. The
    # finite log mass must survive even when FP32 probabilities round to zero.
    model.log_weights.fill_(-1000.)
    model.log_weights[:, 0] = 0.
    prior = model.aggregation_weights()
    assert prior[[0, 2], 1:].eq(0).all()
    assert prior[[1, 3], 1:].gt(0).all()
    torch.testing.assert_close(prior[1, 1:], model.hazard * model.log_prior[1:].exp())
    x = 100 * x
    y = (x @ model.mean_weights()[5:].T).max()
    model.update(x, y)
    recovered = model.aggregation_weights()[2]
    assert recovered[0] < 1e-6
    assert recovered[1:].max() > .1


@torch.no_grad()
def test_compiled_runner_restores_capture_and_consumes_nondivisible_tail():
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
    from cleanrl.plasticity.predictive_mean_risk_eval_v2 import Runner

    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    gen = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(11, 5, device='cuda', generator=gen)
    ys = torch.randn(11, device='cuda', generator=gen)
    args = sparse.Args(input_dim=5, steps=11, graph_steps=4)
    def make():
        return Runner(xs, ys, setup_model(),
                      sparse.LinearLearner('adam', (.001, .003), args, xs, ys))
    actual, reference = make(), make()
    graphs = actual.capture(4)
    for _ in range(2):
        graphs[4].replay()
    for _ in range(3):
        graphs[1].replay()
    for _ in range(11):
        reference.update()
    torch.cuda.synchronize()
    assert int(actual.index) == int(actual.adam.index) == int(actual.adam.steps) == 11
    torch.testing.assert_close(actual.predictions, reference.predictions, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(actual.model.mean_weights(), reference.model.mean_weights(), rtol=3e-4, atol=3e-6)
