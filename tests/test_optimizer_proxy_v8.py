"""Spectral strength, normalized-Adam, and independently tuned head-rate contracts."""

import math

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_model_v6 as old
from cleanrl.plasticity import optimizer_proxy_model_v8 as proxy

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("method,iterations", [("polar_3", 3), ("polar_8", 8)])
def test_spectral_strength_matches_scalar_singular_polynomial(method, iterations):
    singular = torch.tensor([.01, .2, 1.5], device="cuda", dtype=torch.float64)
    q = torch.diag(singular)[None]
    values = [float(s) / (float(singular.norm()) + 1e-7) for s in singular]
    for _ in range(iterations):
        values = [3.4445 * s - 4.775 * s**3 + 2.0315 * s**5 for s in values]
    wanted = torch.diag(singular.new_tensor(values))[None]
    wanted *= .6 / (wanted.norm() + 1e-7)
    torch.testing.assert_close(proxy.matrix_direction(q, method), wanted, rtol=1e-9, atol=1e-10)


def state(dtype=torch.float64):
    gen = torch.Generator(device="cuda").manual_seed(101)
    weights = [torch.randn(3, o, i + 1, generator=gen, device="cuda", dtype=dtype) * .1
               for o, i in [(4, 3), (4, 4), (1, 4)]]
    m = [torch.randn(w.shape, generator=gen, device="cuda", dtype=dtype) * .02 for w in weights]
    v = [torch.rand(w.shape, generator=gen, device="cuda", dtype=dtype) * .1 for w in weights]
    g = [torch.randn(w.shape, generator=gen, device="cuda", dtype=dtype) * .2 for w in weights]
    correction = [torch.randn(w.shape, generator=gen, device="cuda", dtype=dtype) * .01 for w in weights]
    hyper = [weights[0].new_tensor(x).reshape(3, 1, 1) for x in
             ([.001, .003, .007], [.0, .9, .9999], [.95, .999, .9], [.0, .01, .1])]
    step = torch.tensor(5, device="cuda", dtype=torch.int64)
    return weights, [w.clone() for w in weights], m, v, step, g, correction, hyper


@pytest.mark.parametrize("method", ["adamw", "predictive", "mars_1", "polar", "matrix_rms"])
def test_unit_head_ratio_preserves_previous_algorithm(method):
    w, prev, m, v, step, g, c, hyper = state()
    expected = old.transition(w, prev, m, v, step, g, c, *hyper, method, 8)
    actual = proxy.transition(w, prev, m, v, step, g, c, *hyper, method, 8, torch.ones_like(hyper[0]))
    for aa, ee in zip(actual[:3], expected[:3]):
        for a, e in zip(aa, ee):
            torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["adamw", "polar_8", "adam_rms"])
def test_compiled_head_lr_changes_only_final_group_with_correct_decay(method):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    w, prev, m, v, step, g, c, hyper = state(torch.float32)
    scale = w[0].new_tensor([.25, 2., 4.]).reshape(3, 1, 1)
    compiled = torch.compile(proxy.transition, fullgraph=True, options={"triton.cudagraphs": False})
    actual = compiled(w, prev, m, v, step, g, c, *hyper, method, 8, scale)
    unit = proxy.transition(w, prev, m, v, step, g, c, *hyper, method, 8, torch.ones_like(scale))
    for a, e in zip(actual[0][:-1], unit[0][:-1]):
        torch.testing.assert_close(a, e, rtol=3e-4, atol=3e-6)
    for k in range(3):
        lr, b1, b2, decay = [float(x[k]) for x in hyper]
        head_lr = lr * float(scale[k])
        first = b1 * m[-1][k].double() + (1-b1) * g[-1][k].double()
        second = b2 * v[-1][k].double() + (1-b2) * g[-1][k].double().square()
        expected = w[-1][k].double().clone()
        expected[..., :-1] *= 1-head_lr*decay
        expected -= head_lr*(first/(1-b1**6))/((second/(1-b2**6)).sqrt()+1e-8)
        torch.testing.assert_close(actual[0][-1][k].double(), expected, rtol=3e-4, atol=3e-6)


def test_normalized_adam_control_retains_diagonal_direction_not_raw_momentum():
    w, prev, m, v, step, g, c, hyper = state()
    result = proxy.transition(w, prev, m, v, step, g, c, *hyper, "adam_rms", 8, torch.ones_like(hyper[0]))
    lr, b1, b2, decay = hyper
    for i in range(2):
        first = b1*m[i]+(1-b1)*g[i]
        second = b2*v[i]+(1-b2)*g[i].square()
        q = (first/(1-b1.pow(6)))/((second/(1-b2.pow(6))).sqrt()+1e-8)
        q = q[..., :-1]
        direction = q * (.2*math.sqrt(q.shape[-1]*q.shape[-2])) / (q.norm(dim=(-1,-2),keepdim=True)+1e-7)
        wanted = w[i][..., :-1]*(1-lr*decay)-lr*direction
        torch.testing.assert_close(result[0][i][..., :-1], wanted, rtol=1e-10, atol=1e-12)
