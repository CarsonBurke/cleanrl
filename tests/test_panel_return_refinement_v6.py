"""New beta2 boundary contracts; execute through mlq on CUDA."""

import pytest
import torch

from cleanrl.plasticity.panel_return_representation_v5 import Config, Learner
from cleanrl.plasticity import panel_return_refinement_eval_v6 as evaluation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("beta2", [.5, .9999])
def test_refined_beta2_boundaries_match_independent_adam_trajectory(beta2):
    generator = torch.Generator(device="cuda").manual_seed(51)
    x = torch.randn((7, 5), generator=generator, device="cuda")
    frames = (x, x, torch.cat((x, torch.zeros((7, 24), device="cuda")), dim=1))
    y = torch.randn(7, generator=generator, device="cuda")
    learner = Learner(5, 128, (Config("memory_adam", .003, beta2),), "cuda", num_samples=7)
    parameters = [p[0].detach().clone().requires_grad_() for p in learner.parameters]
    optimizer = torch.optim.Adam(parameters, lr=.003, betas=(.9, beta2), eps=1e-8, foreach=False)
    for step in range(5):
        mask = torch.arange(7, device="cuda") != step
        w1, b1, w2, b2, w3, b3 = parameters
        hidden = torch.tanh(torch.tanh(frames[2] @ w1.T + b1) @ w2.T + b2)
        forecast = (hidden @ w3.T + b3).squeeze(-1)
        target = y + .1 * step
        actual = learner.step(frames, torch.where(mask, target, float("nan")), target.square(), mask).clone()
        torch.testing.assert_close(actual[0], forecast.detach(), rtol=5e-5, atol=3e-6)
        optimizer.zero_grad(set_to_none=True)
        (forecast[mask] - target[mask]).square().mean().backward()
        optimizer.step()
        for actual_parameter, reference in zip(learner.parameters, parameters):
            torch.testing.assert_close(actual_parameter[0], reference, rtol=5e-5, atol=3e-6)


def test_refinement_requires_its_declared_grid_before_experiment():
    # A v5 default would silently run the old lower-bound search again.
    with pytest.raises(ValueError, match="LR grid"):
        evaluation.validate_args(evaluation.Args(output_dir="unused", lrs=(3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)))
