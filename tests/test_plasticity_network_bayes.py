"""Full EKF posterior and direction-erasure contracts; execute CUDA through mlq."""

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as bayes
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def linear_learner(method="network"):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = bayes.Args(input_dim=1, hidden=1, diffusion=0, known_noise=True, graph_steps=2)
    initial = [torch.tensor(value, device="cuda") for value in (
        [[1.1, 0.2]], [[0.7, -0.3]], [[0.25, -0.1]])]
    xs = torch.tensor([[-1.2], [0.4], [1.1], [-0.2]], device="cuda")
    ys = torch.tensor([1.7, -0.4, 2.2, 0.9], device="cuda")
    noise = torch.tensor([0.4, 1.2, 0.7, 0.9], device="cuda")
    learner = bayes.Learner(method, (2.0,), initial, a, xs, ys, ys, noise)
    # Known hidden weights make the actual network exactly linear in the two
    # output parameters. Zero hidden covariance freezes them, without a fake model.
    learner.cov.zero_()
    learner.cov[0, -2:, -2:].copy_(torch.tensor([[1.2, 0.35], [0.35, 0.8]], device="cuda"))
    return learner


@torch.no_grad()
def test_correlated_two_parameter_gaussian_posterior_and_predictive_uncertainty():
    learner = linear_learner()
    projected = linear_learner()
    hidden = [w.clone() for w in learner.weights[:2]]
    h2 = bayes.forward(learner.weights, learner.xs)[1][0, :, 0].double()
    design = torch.stack((h2, torch.ones_like(h2)), -1)
    prior_precision = torch.linalg.inv(learner.cov[0, -2:, -2:].double())
    prior_mean = learner.weights[-1][0, 0].double().clone()
    for n in range(1, len(learner.xs) + 1):
        learner.update()
        # A deliberately diagonal posterior loses information about correlated
        # features. This is a counterfactual, not another implementation of EKF.
        projected.cov.copy_(torch.diag_embed(projected.cov.diagonal(dim1=-2, dim2=-1)))
        projected.update()
        x = design[:n]
        precision = prior_precision + x.T @ (x / learner.noise_var[:n, None].double())
        covariance = torch.linalg.inv(precision)
        mean = torch.linalg.solve(precision, prior_precision @ prior_mean
                                 + x.T @ (learner.ys[:n] / learner.noise_var[:n]).double())
        torch.testing.assert_close(learner.weights[-1][0, 0], mean.float(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(learner.cov[0, -2:, -2:], covariance.float(), rtol=2e-5, atol=2e-6)
        prediction = bayes.forward(learner.weights, learner.xs)[2][0]
        torch.testing.assert_close(prediction, (design @ mean).float(), rtol=2e-5, atol=2e-6)
        predictive_variance = torch.einsum("bi,ij,bj->b", design,
                                           learner.cov[0, -2:, -2:].double(), design)
        torch.testing.assert_close(predictive_variance, (design @ covariance * design).sum(-1),
                                   rtol=2e-5, atol=2e-6)
    for actual, fixed in zip(learner.weights[:2], hidden):
        torch.testing.assert_close(actual, fixed, rtol=0, atol=0)
    projected_prediction = bayes.forward(projected.weights, learner.xs)[2][0]
    assert (projected_prediction - prediction).abs().max().item() > 1e-3


@torch.no_grad()
def test_cross_neuron_covariance_transfers_credit_to_zero_local_jacobian():
    learner = linear_learner()
    # Zero output weight means hidden weights have zero local Jacobian. A
    # correlation between first-layer bias and output bias must nevertheless
    # move that hidden bias. A block/diagonal approximation cannot do so.
    learner.weights[-1][0, 0, 0] = 0
    learner.cov.zero_()
    learner.cov[0, 1, 1] = 1
    learner.cov[0, -1, -1] = 1
    learner.cov[0, 1, -1] = 0.6
    learner.cov[0, -1, 1] = 0.6
    before = learner.weights[0][0, 0, -1].clone()
    residual = learner.weights[-1][0, 0, -1] - learner.ys[0]
    expected = before - 0.6 * residual / (learner.noise_var[0] + 1)
    learner.update()
    torch.testing.assert_close(learner.weights[0][0, 0, -1], expected)
    torch.testing.assert_close(learner.cov[0, 1, -1],
                               0.6 * learner.noise_var[0] / (learner.noise_var[0] + 1))
    assert not torch.equal(learner.weights[0][0, 0, -1], before)


@torch.no_grad()
def test_scalar_erases_mean_direction_but_retains_full_conditioning_and_diagonal_process():
    full, scalar = linear_learner(), linear_learner("network_scalar")
    # A nonzero, nonuniform Q catches accidental scalar broadcast or omission.
    process = torch.linspace(0.01, 0.06, full.cov.shape[-1], device="cuda").unsqueeze(0)
    full.process.copy_(process)
    scalar.process.copy_(process)
    before = torch.cat([w.flatten(1) for w in full.weights], -1)
    _, inputs, sensitivities = bayes.sample_state(full.weights, full.xs[0])
    jacobian = torch.cat([(j.unsqueeze(-1) * inp.unsqueeze(1)).flatten(1)
                          for inp, j in zip(inputs, sensitivities)], -1)
    prior = full.cov.clone() + torch.diag_embed(process)
    pj = (prior @ jacobian.unsqueeze(-1)).squeeze(-1)
    leverage = (jacobian * pj).sum(-1)
    denominator = full.noise_var[0] + leverage
    residual = bayes.sample_state(full.weights, full.xs[0])[0] - full.ys[0]
    expected_cov = prior - pj.unsqueeze(-1) * pj.unsqueeze(-2) / denominator[:, None, None]
    full.update()
    scalar.update()
    torch.testing.assert_close(full.cov, expected_cov)
    torch.testing.assert_close(scalar.cov, expected_cov)
    delta_full = torch.cat([w.flatten(1) for w in full.weights], -1) - before
    delta_scalar = torch.cat([w.flatten(1) for w in scalar.weights], -1) - before
    alpha = leverage / (denominator * jacobian.square().sum(-1))
    assert (alpha > 0).all().item()
    torch.testing.assert_close(delta_scalar, -residual[:, None] * alpha[:, None] * jacobian,
                               rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(delta_full, -residual[:, None] * pj / denominator[:, None],
                               rtol=2e-5, atol=2e-7)
    assert (delta_full - delta_scalar).abs().max().item() > 1e-3


@torch.no_grad()
@pytest.mark.parametrize("method", ["network", "network_scalar"])
def test_captured_updates_preserve_stream_position_and_posterior(method):
    actual, eager = linear_learner(method), linear_learner(method)
    graph, _ = actual.capture()
    # Capture must leave the next real observation unconsumed. Replaying the
    # entire available stream catches both missing rollback and double updates.
    for _ in range(len(actual.xs) // actual.capture_steps):
        graph.replay()
        for _ in range(eager.capture_steps):
            eager.update()
    torch.cuda.synchronize()
    for observed, expected in zip(actual.weights, eager.weights):
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.cov, eager.cov, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.error, eager.error, rtol=2e-5, atol=2e-6)
    assert actual.index.item() == len(actual.xs)
