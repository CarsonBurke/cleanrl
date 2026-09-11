"""Matrix geometry controls and condition-aware compiled transition audits."""

import math

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_model_v4 as adam
from cleanrl.plasticity import optimizer_proxy_model_v6 as proxy

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("rows,cols", [(3, 5), (5, 3), (4, 4)])
def test_polar_matches_independent_singular_value_polynomial(rows, cols):
    gen = torch.Generator(device="cuda").manual_seed(22)
    u = torch.linalg.qr(torch.randn(rows, rows, generator=gen, device="cuda", dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(cols, cols, generator=gen, device="cuda", dtype=torch.float64))[0]
    rank = min(rows, cols)
    singular = torch.linspace(.2, 2., rank, device="cuda", dtype=torch.float64)
    matrix = (u[:, :rank] * singular) @ v[:, :rank].T
    expected_s = [float(s) / (float(singular.norm()) + 1e-7) for s in singular]
    for _ in range(5):
        expected_s = [3.4445*s - 4.775*s**3 + 2.0315*s**5 for s in expected_s]
    s = singular.new_tensor(expected_s)
    wanted = (u[:, :rank] * s) @ v[:, :rank].T
    wanted *= (.2 * math.sqrt(rows * cols)) / (wanted.norm() + 1e-7)
    actual = proxy.matrix_direction(matrix[None], "polar")[0]
    torch.testing.assert_close(actual, wanted, rtol=1e-9, atol=1e-10)
    rms = proxy.matrix_direction(matrix[None], "matrix_rms")[0]
    expected_rms = matrix * (.2 * math.sqrt(rows * cols)) / (matrix.norm() + 1e-7)
    torch.testing.assert_close(rms, expected_rms, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(actual.norm(), rms.norm(), rtol=2e-7, atol=1e-8)
    # Same norm, genuinely different matrix directions for this unequal spectrum.
    assert (actual - rms).norm() > .05
    for method in ("polar", "matrix_rms"):
        torch.testing.assert_close(proxy.matrix_direction(torch.zeros_like(matrix)[None], method),
                                   torch.zeros_like(matrix)[None], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["polar", "matrix_rms"])
def test_compiled_polar_step_preserves_adamw_head_and_bias_contract(method):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(72)
    weights = [torch.randn(3, o, i + 1, generator=gen, device="cuda") * .2
               for o, i in [(4, 3), (4, 4), (1, 4)]]
    previous = [w.clone() for w in weights]
    m = [torch.randn(w.shape, generator=gen, device="cuda") * .1 for w in weights]
    v = [torch.rand(w.shape, generator=gen, device="cuda") * .3 for w in weights]
    g = [torch.randn(w.shape, generator=gen, device="cuda") * .1 for w in weights]
    corrections = [torch.zeros_like(w) for w in weights]
    hyper = [weights[0].new_tensor(a).reshape(3, 1, 1) for a in
             ([.003, .007, .02], [0., .9, .999], [.95, .999, .9], [.0, .03, .1])]
    step = torch.tensor(3, device="cuda", dtype=torch.int64)
    compiled = torch.compile(proxy.transition, fullgraph=True, options={"triton.cudagraphs": False})
    result = compiled(weights, previous, m, v, step, g, corrections, *hyper, method, 8)
    w64, prev64, m64, v64, g64, c64 = [[x.double() for x in tensors]
                                     for tensors in (weights, previous, m, v, g, corrections)]
    h64 = [x.double() for x in hyper]
    oracle = adam.transition(w64, prev64, m64, v64, step, g64, c64, *h64, "adamw")
    lr, beta1, _, decay = h64
    mass = 1 - beta1.pow(4)
    for layer, (actual, wanted) in enumerate(zip(result[0], oracle[0])):
        if layer == len(weights) - 1:
            torch.testing.assert_close(actual.double(), wanted, rtol=3e-4, atol=3e-6)
        else:
            torch.testing.assert_close(actual[..., -1].double(), wanted[..., -1], rtol=3e-4, atol=3e-6)
            q = beta1 * (oracle[1][layer] / mass) + (1-beta1) * g64[layer]
            direction = proxy.matrix_direction(q[..., :-1], method)
            expected = w64[layer][..., :-1] * (1-lr*decay) - lr*direction
            torch.testing.assert_close(actual[..., :-1].double(), expected, rtol=3e-4, atol=3e-6)
    assert int(result[3]) == 4


def test_transition_audit_allows_input_rounding_not_missing_or_corrupt_updates():
    from cleanrl.plasticity.optimizer_proxy_eval_v6 import RoundLearner

    before = [torch.tensor([1000., .1], device="cuda")]
    eager = [torch.tensor([0., .099], device="cuda")]
    # One cancellation ULP at the scale of the original input, not the tiny result.
    compiled = [eager[0].clone()]
    compiled[0][0] = 2**-14
    RoundLearner.audit_transition(before, compiled, eager)
    with pytest.raises(AssertionError):
        RoundLearner.audit_transition(before, before, eager)
    corrupted = [eager[0].clone()]
    corrupted[0][1] += .01
    with pytest.raises(AssertionError):
        RoundLearner.audit_transition(before, corrupted, eager)
    nonfinite = [eager[0].clone()]
    nonfinite[0][0] = float("nan")
    with pytest.raises(AssertionError):
        RoundLearner.audit_transition(before, nonfinite, eager)
    with pytest.raises(AssertionError):
        RoundLearner.audit_transition(before, eager, nonfinite)
