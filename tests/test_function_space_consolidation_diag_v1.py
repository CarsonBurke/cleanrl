"""Contracts for the function-space diagnostic: exact batched Jacobian and an idempotent row-space projection."""

import pytest
import torch

from cleanrl.plasticity import function_space_consolidation_diag_v1 as diag
from cleanrl.plasticity import network_bayes_stream_v2 as reference

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def make():
    gen = torch.Generator(device="cuda").manual_seed(3)
    weights = [torch.randn(2, o, i + 1, generator=gen, device="cuda", dtype=torch.float64) * .3
               for o, i in [(5, 3), (5, 5), (1, 5)]]
    x = torch.randn(4, 3, generator=gen, device="cuda", dtype=torch.float64)
    return weights, x


def test_batched_jacobian_matches_autograd():
    weights, x = make()
    jac = diag.jacobian(weights, x)
    for k in range(2):
        for b in range(4):
            leaves = [w[k].clone().requires_grad_(True) for w in weights]
            out = reference.forward([leaf.unsqueeze(0) for leaf in leaves], x[b:b + 1])[2].sum()
            grads = torch.autograd.grad(out, leaves)
            expected = torch.cat([g.flatten() for g in grads])
            torch.testing.assert_close(jac[k, b], expected, rtol=1e-10, atol=1e-12)


def test_projection_is_idempotent_and_preserves_first_order_output_change():
    weights, x = make()
    jac = diag.jacobian(weights, x)
    d = torch.randn(2, jac.shape[-1], device="cuda", dtype=torch.float64)
    par = diag.project(jac, d)
    # Relative damping 1e-8 on the Gram diagonal bounds the departure from an exact projector.
    torch.testing.assert_close(diag.project(jac, par), par, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(jac @ par.unsqueeze(-1), jac @ d.unsqueeze(-1), rtol=1e-4, atol=1e-6)
    perp = d - par
    assert float((jac @ perp.unsqueeze(-1)).abs().max()) < 1e-5
    assert float((par * perp).sum(-1).abs().max()) < 1e-5
