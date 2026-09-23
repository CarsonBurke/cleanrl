"""Contracts for v11: shrunk matrix families reproduce their parents at zero variance, shrink with relative
variance, pass every v10 family through bit-exactly, and the structured regime's teacher is distractor-invariant."""

import math

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import optimizer_proxy_eval_v11 as evaluation
from cleanrl.plasticity import optimizer_proxy_model_v10 as prior
from cleanrl.plasticity import optimizer_proxy_model_v11 as proxy
from tests.test_optimizer_proxy_v9 import grads_like, state

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def roll(method, steps, grads_fn, transition, dtype=torch.float64):
    w, prev, m, v, hyper = state(dtype)
    m = [torch.zeros_like(x) for x in m]
    v = [torch.zeros_like(x) for x in v]
    aux = proxy.initial_aux(w)
    step = torch.tensor(0, device="cuda", dtype=torch.int64)
    for t in range(steps):
        grads, corrections = grads_fn(w, t)
        w, m, v, aux, step = transition(w, prev, m, v, aux, step, grads, corrections, hyper, method, 1, 32)
    return w, m, v


def constant_gradients(seed):
    cache = {}

    def draw(w, t):
        if "g" not in cache:
            cache["g"] = grads_like(w, seed)
        return cache["g"]
    return draw


@pytest.mark.parametrize("method,parent", [("polar_svag", "polar"), ("rms_svag", "matrix_rms")])
def test_zero_variance_stream_reproduces_parent_weights_bit_exactly(method, parent):
    # beta1 = 0 is excluded from shrunk grids; use momentum candidates only (indices 1, 2 of the v9 state).
    w_new, _, v_new = roll(method, 6, constant_gradients(7), proxy.transition)
    w_old, _, v_old = roll(parent, 6, constant_gradients(7), proxy.transition)
    for a, b in zip(w_new, w_old):
        torch.testing.assert_close(a[1:], b[1:], rtol=0, atol=0)
    # The shrunk family keeps a full hidden second moment; the parent keeps only the bias column.
    assert torch.count_nonzero(v_old[0][..., :-1]) == 0 and torch.count_nonzero(v_new[0][..., :-1]) > 0


def test_shrinkage_is_one_at_zero_variance_and_decreasing_in_relative_variance():
    gen = torch.Generator(device="cuda").manual_seed(3)
    # One momentum shared by the three candidates, so only beta1 (through rho) differs between them.
    m = torch.randn(1, 4, 5, generator=gen, device="cuda", dtype=torch.float64).expand(3, 4, 5).clone()
    beta1 = torch.tensor([.5, .9, .99], device="cuda", dtype=torch.float64).reshape(3, 1, 1)
    count = torch.tensor(40, device="cuda")
    torch.testing.assert_close(proxy.shrinkage(m, m.square(), beta1, count), torch.ones_like(m), rtol=0, atol=0)
    # Ulp-level excess from a different rounding path must not move gamma off one either.
    torch.testing.assert_close(proxy.shrinkage(m, m.square() * (1 + 2 * torch.finfo(m.dtype).eps), beta1, count),
                               torch.ones_like(m), rtol=0, atol=0)
    zero = torch.zeros_like(beta1)
    torch.testing.assert_close(proxy.shrinkage(m, m.square() + 1.0, zero, count), torch.ones_like(m), rtol=0, atol=0)
    low = proxy.shrinkage(m, m.square() + 0.1, beta1, count)
    high = proxy.shrinkage(m, m.square() + 1.0, beta1, count)
    assert bool((high < low).all()) and bool((high > 0).all()) and bool((low < 1).all())
    # Scale-free: rescaling m and v together leaves gamma unchanged.
    torch.testing.assert_close(proxy.shrinkage(3 * m, 9 * (m.square() + 1.0), beta1, count), high, rtol=1e-12, atol=0)
    # Longer momentum (smaller rho) shrinks less at the same relative variance.
    assert bool((high[2] > high[1]).all()) and bool((high[1] > high[0]).all())


def test_variance_reduction_matches_closed_form():
    beta1 = torch.tensor([.9], device="cuda", dtype=torch.float64).reshape(1, 1, 1)
    t = 12
    expected = (1 - .9) * (1 + .9 ** (t + 1)) / ((1 + .9) * (1 - .9 ** (t + 1)))
    torch.testing.assert_close(proxy.variance_reduction(beta1, torch.tensor(t, device="cuda")),
                               torch.full_like(beta1, expected), rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", ["adamw", "polar", "matrix_rms", "look_polar", "tier_adamw"])
def test_v10_families_pass_through_bit_exactly(method):
    def noisy(w, t):
        return grads_like(w, 100 + t)
    new = roll(method, 5, noisy, proxy.transition)
    old = roll(method, 5, noisy, prior.transition)
    for group_new, group_old in zip(new, old):
        for a, b in zip(group_new, group_old):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_methods_and_gradient_routing():
    assert set(prior.METHODS) < set(proxy.METHODS) and {"polar_svag", "rms_svag"} <= set(proxy.METHODS)
    w, prev, _, _, _ = state()
    x = torch.randn(4, 3, device="cuda", dtype=torch.float64)
    y = torch.randn(4, device="cuda", dtype=torch.float64)
    zeros = torch.zeros(3, 4, device="cuda", dtype=torch.float64)
    for method, parent in (("polar_svag", "polar"), ("rms_svag", "matrix_rms")):
        g_new, _ = proxy.gradients(w, prev, x, y, zeros, zeros, zeros, "regression", method)
        g_old, _ = proxy.gradients(w, prev, x, y, zeros, zeros, zeros, "regression", parent)
        for a, b in zip(g_new, g_old):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_structured_teacher_and_off_marginal_sets():
    class A:
        hidden, input_dim = 64, 17
    gen = torch.Generator(device="cuda").manual_seed(2)
    teacher = reference.draw_teacher(A, gen, "cuda")
    original = [layer.clone() for layer in teacher]
    structured = evaluation.structure_teacher(teacher, 4, 17)
    assert torch.count_nonzero(structured[0][:, 4:]) == 0
    torch.testing.assert_close(structured[0][:, :4], original[0][:, :4] * math.sqrt(17 / 4), rtol=0, atol=0)
    for layer, before in zip(teacher, original):
        torch.testing.assert_close(layer, before, rtol=0, atol=0)  # input teacher untouched
    scenario = {"relevant": 4, "distractor_scale": 3.0, "relevant_scale": 1.5}
    x = torch.randn(512, 17, generator=gen, device="cuda")
    before = x.clone()
    sets = evaluation.off_marginal(x, scenario)
    torch.testing.assert_close(reference.teach(structured, sets["distractor"]), reference.teach(structured, x), rtol=0, atol=0)
    torch.testing.assert_close(sets["relevant"][:, 4:], x[:, 4:], rtol=0, atol=0)
    assert not torch.allclose(reference.teach(structured, sets["relevant"]), reference.teach(structured, x))
    torch.testing.assert_close(x, before, rtol=0, atol=0)  # inputs untouched


def test_v11_plan_grids_are_lean_and_exclude_zero_momentum_for_shrunk_families():
    import json
    from pathlib import Path
    plan = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v11_plan.json").read_text())
    names = [s["name"] for s in plan["suite"]]
    assert "structured_ood" in names and set(plan["families"]) == {"adamw", "polar", "matrix_rms", "polar_svag", "rms_svag"}
    for family in plan["families"]:
        for scenario in names:
            configs = evaluation.family_grid(plan, family, scenario)
            assert 13 * 4 * 2 <= len(configs) <= 13 * 4 * 3
            if family.endswith("_svag"):
                assert all(config["beta1"] > 0 for config in configs)
