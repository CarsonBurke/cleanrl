"""Exact correlated-quadratic contracts; run on CUDA via mlq, not a benchmark."""

import pytest
import torch

from cleanrl.plasticity.predictive_covariance_optimizer_v2 import CovarianceState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def observations():
    x = torch.tensor([[1., 2., -1.], [2., 3., 0.5], [-1., -2., 2.],
                      [0.5, 0.7, -1.], [3., 4., 1.]], device="cuda")
    y = torch.tensor([1., 0.5, -0.8, 0.3, 1.2], device="cuda")
    return x, y


def test_dense_posterior_matches_correlated_batch_solution():
    x, y = observations()
    anchor = torch.tensor([[0.2, -0.1, 0.3], [0.2, -0.1, 0.3]], device="cuda")
    weight = anchor.clone()
    scales = torch.tensor([[0.3], [2.0]], device="cuda")
    state = CovarianceState(weight, prior_scale=scales, prior_density=1.0)
    compiled = torch.compile(state.step, fullgraph=True, options={"triton.cudagraphs": False})
    for xt, yt in zip(x, y):
        residual = (weight * xt).sum(1) - yt
        compiled(weight, residual[:, None] * xt, xt.square()[None, :],
                 residual[:, None].square(), features=xt, targets=yt.expand(2))
    for i in range(2):
        xd = x.double()
        precision = xd.T @ xd + torch.eye(3, device="cuda", dtype=torch.float64) / scales[i].double().square()
        response = y.double() - xd @ anchor[i].double()
        expected = anchor[i].double() + torch.linalg.solve(precision, xd.T @ response)
        torch.testing.assert_close(weight[i].double(), expected, rtol=2e-6, atol=2e-7)


def test_final_predictor_is_independent_of_observation_order():
    x, y = observations()
    weights = [torch.zeros((2, 3), device="cuda") for _ in range(2)]
    states = [CovarianceState(w, prior_scale=0.7,
                              prior_density=torch.tensor([[0.1], [1.0]], device="cuda"))
              for w in weights]
    orders = [range(len(x)), (4, 2, 0, 3, 1)]
    for weight, state, order in zip(weights, states, orders):
        for i in order:
            residual = (weight * x[i]).sum(1) - y[i]
            state.step(weight, residual[:, None] * x[i], x[i].square()[None, :],
                       residual[:, None].square(), features=x[i], targets=y[i].expand(2))
    torch.testing.assert_close(weights[0], weights[1], rtol=2e-6, atol=2e-7)


def test_zero_outcomes_preserve_zero_predictions_with_correlated_inputs():
    x, _ = observations()
    weight = torch.zeros((2, 3), device="cuda")
    state = CovarianceState(weight, prior_scale=10.,
                            prior_density=torch.tensor([[0.01], [1.0]], device="cuda"))
    for xt in x:
        state.step(weight, torch.zeros_like(weight), xt.square()[None, :],
                   torch.zeros((2, 1), device="cuda"), features=xt,
                   targets=torch.zeros(2, device="cuda"))
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    assert all(torch.isfinite(buffer).all() for buffer in state.buffers())
