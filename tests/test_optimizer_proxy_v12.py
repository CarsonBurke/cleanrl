"""Contracts for v12: gated families reproduce their parents at decay 0 bit-exactly and at gate 1 up to
rounding order, the gate is a calibrated probability with exact zero-variance limits, the EMAs live in
aux slots 2/3, every v11 family passes through bit-exactly, and the plan carries the gate axis."""

import json
import math
from pathlib import Path

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_eval_v12 as evaluation
from cleanrl.plasticity import optimizer_proxy_model_v11 as prior
from cleanrl.plasticity import optimizer_proxy_model_v12 as proxy
from tests.test_optimizer_proxy_v9 import grads_like, state

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

PAIRS = [("gate_adamw", "adamw"), ("gate_polar", "polar")]


def gated_state(dtype=torch.float64, k=3, floor=0.0):
    """v9 test state plus the gate axis; `floor` keeps every weight away from zero (sign-stable)."""
    w, prev, m, v, hyper = state(dtype, k)
    if floor:
        w = [torch.sign(x) * (x.abs() + floor) for x in w]
        prev = [x.clone() for x in w]
    hyper = {**hyper, "gate_beta": w[0].new_tensor([.99, .999, .9]).reshape(k, 1, 1)}
    return w, prev, m, v, hyper


def roll(method, steps, grads_fn, transition, hyper_override=None, floor=0.0):
    w, prev, m, v, hyper = gated_state(floor=floor)
    hyper = {**hyper, **(hyper_override or {})}
    m = [torch.zeros_like(x) for x in m]
    v = [torch.zeros_like(x) for x in v]
    aux = proxy.initial_aux(w)
    step = torch.tensor(0, device="cuda", dtype=torch.int64)
    for t in range(steps):
        grads, corrections = grads_fn(w, t)
        w, m, v, aux, step = transition(w, prev, m, v, aux, step, grads, corrections, hyper, method, 1, 32)
    return w, m, v, aux


def noisy(w, t):
    return grads_like(w, 100 + t)


@pytest.mark.parametrize("method,parent", PAIRS)
def test_decay_zero_reproduces_parent_bit_exactly(method, parent):
    zero = {"weight_decay": torch.zeros(3, 1, 1, device="cuda", dtype=torch.float64)}
    new = roll(method, 6, noisy, proxy.transition, zero)
    old = roll(parent, 6, noisy, proxy.transition, zero)
    for a, b in zip(new[0], old[0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("method,parent", PAIRS)
def test_shrink_everywhere_stream_equals_parent_at_full_decay(method, parent):
    """A zero-variance gradient stream that agrees with shrinking on every coordinate: gate exactly one."""
    cache = {}

    def shrinking(w, t):
        if "g" not in cache:
            cache["g"] = [0.2 * torch.sign(x) for x in w]  # constant, same sign as the (sign-stable) weights
        return cache["g"], [torch.zeros_like(x) for x in cache["g"]]
    new = roll(method, 6, shrinking, proxy.transition, floor=0.5)
    old = roll(parent, 6, shrinking, proxy.transition, floor=0.5)
    for a, b in zip(new[0], old[0]):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    # ... and a stream that fights the decay everywhere: gate exactly zero, equal to the parent at decay 0.
    cache.clear()

    def growing(w, t):
        if "g" not in cache:
            cache["g"] = [-0.2 * torch.sign(x) for x in w]
        return cache["g"], [torch.zeros_like(x) for x in cache["g"]]
    zero = {"weight_decay": torch.zeros(3, 1, 1, device="cuda", dtype=torch.float64)}
    new = roll(method, 6, growing, proxy.transition, floor=0.5)
    old = roll(parent, 6, growing, proxy.transition, zero, floor=0.5)
    for a, b in zip(new[0], old[0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resistance_gate_is_a_calibrated_probability():
    gen = torch.Generator(device="cuda").manual_seed(4)
    # One mean and one weight tensor shared by the three candidates, so only beta (through rho) differs between them.
    s = torch.randn(1, 4, 5, generator=gen, device="cuda", dtype=torch.float64).expand(3, 4, 5).clone()
    w = torch.randn(1, 4, 5, generator=gen, device="cuda", dtype=torch.float64).expand(3, 4, 5).clone()
    beta = torch.tensor([.9, .99, .999], device="cuda", dtype=torch.float64).reshape(3, 1, 1)
    count = torch.tensor(500, device="cuda")
    # Zero variance: exactly one where s and w share a sign, exactly zero where they do not.
    gate = proxy.resistance_gate(s, s.square(), w, beta, count)
    torch.testing.assert_close(gate, (torch.sign(s) * torch.sign(w) > 0).to(gate.dtype), rtol=0, atol=0)
    # Zero mean: exactly one half.
    torch.testing.assert_close(proxy.resistance_gate(torch.zeros_like(s), torch.ones_like(s), w, beta, count),
                               torch.full_like(s, 0.5), rtol=0, atol=0)
    # Closed form: Phi(s sign(w) / sqrt(rho q / (1 - rho))) with q the excess variance.
    q = s.square() + 2.0
    rho = proxy.variance_reduction(beta, count)
    z = s * torch.sign(w) / torch.sqrt(rho * 2.0 / (1 - rho))
    expected = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    torch.testing.assert_close(proxy.resistance_gate(s, q, w, beta, count), expected, rtol=1e-10, atol=1e-12)
    # Scale-free in the gradient, monotone in the mean, and more decisive with a slower pole (smaller rho).
    torch.testing.assert_close(proxy.resistance_gate(3 * s, 9 * q, w, beta, count), expected, rtol=1e-10, atol=1e-12)
    stronger = proxy.resistance_gate(2 * s, 4 * s.square() + 2.0, w, beta, count)
    assert bool(((stronger - 0.5).abs() >= (expected - 0.5).abs() - 1e-12).all())
    assert bool(((expected[2] - 0.5).abs() >= (expected[1] - 0.5).abs() - 1e-12).all())


def test_variance_reduction_counts_updates_not_updates_plus_one():
    beta = torch.tensor([.9], device="cuda", dtype=torch.float64).reshape(1, 1, 1)
    n = 12
    expected = (1 - .9) * (1 + .9 ** n) / ((1 + .9) * (1 - .9 ** n))
    torch.testing.assert_close(proxy.variance_reduction(beta, torch.tensor(n, device="cuda")),
                               torch.full_like(beta, expected), rtol=1e-12, atol=0)
    # One update: the corrected EMA is the sample itself, rho = 1 exactly.
    torch.testing.assert_close(proxy.variance_reduction(beta, torch.tensor(1, device="cuda")), torch.ones_like(beta), rtol=0, atol=0)


def test_gate_state_lives_in_aux_slots_two_and_three():
    cache = {}

    def constant(w, t):
        if "g" not in cache:
            cache["g"] = grads_like(w, 11)
        return cache["g"]
    steps = 7
    _, _, _, aux = roll("gate_polar", steps, constant, proxy.transition)
    w0, _, _, _, hyper = gated_state()
    beta = hyper["gate_beta"]
    for state_layer, g_layer, w_layer in zip(aux, cache["g"][0], w0):
        torch.testing.assert_close(state_layer[:, proxy.MEAN_SLOT], g_layer * (1 - beta ** steps), rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(state_layer[:, proxy.POWER_SLOT], g_layer.square() * (1 - beta ** steps), rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(state_layer[:, 0], w_layer, rtol=0, atol=0)  # phi untouched
        torch.testing.assert_close(state_layer[:, 1], torch.ones_like(w_layer), rtol=0, atol=0)
        assert torch.count_nonzero(state_layer[:, 4:]) == 0


@pytest.mark.parametrize("method", ["adamw", "polar", "polar_svag", "look_polar", "tier_adamw"])
def test_v11_families_pass_through_bit_exactly(method):
    new = roll(method, 5, noisy, proxy.transition)
    old = roll(method, 5, noisy, prior.transition)
    for group_new, group_old in zip(new, old):
        for a, b in zip(group_new, group_old):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_methods_and_gradient_routing():
    assert set(prior.METHODS) < set(proxy.METHODS) and {"gate_adamw", "gate_polar"} <= set(proxy.METHODS)
    assert all(proxy.is_gated(m) for m, _ in PAIRS) and not proxy.is_gated("polar") and not proxy.is_tier("gate_polar")
    w, prev, _, _, _ = gated_state()
    x = torch.randn(4, 3, device="cuda", dtype=torch.float64)
    y = torch.randn(4, device="cuda", dtype=torch.float64)
    zeros = torch.zeros(3, 4, device="cuda", dtype=torch.float64)
    for method, parent in PAIRS:
        g_new, _ = proxy.gradients(w, prev, x, y, zeros, zeros, zeros, "regression", method)
        g_old, _ = proxy.gradients(w, prev, x, y, zeros, zeros, zeros, "regression", parent)
        for a, b in zip(g_new, g_old):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_learner_requires_the_gate_axis_for_gated_families():
    gen = torch.Generator(device="cuda").manual_seed(1)
    initial = [torch.randn(o, i + 1, generator=gen, device="cuda") * .1 for o, i in [(4, 3), (4, 4), (1, 4)]]
    scenario = {"name": "t", "objective": "regression", "samples": 8, "batch_size": 1, "epochs": 1, "drift": False, "block": 1}
    config = {"lr": 1e-3, "beta1": .9, "beta2": .99, "weight_decay": .1, "head_lr_scale": 1.0}
    with pytest.raises(AssertionError, match="gate axis"):
        evaluation.RoundLearner(initial, [config], scenario, "gate_polar")
    learner = evaluation.RoundLearner(initial, [{**config, "gate_beta": .99}], scenario, "gate_polar")
    assert "gate_beta" in learner.hyper and learner.hyper["gate_beta"].shape == (1, 1, 1)


def test_v12_plan_grids():
    plan = json.loads(Path("benchmarks/plasticity/optimizer_proxy_v12_plan.json").read_text())
    names = [s["name"] for s in plan["suite"]]
    assert set(names) == {"iid_online", "drifting_reuse", "clipped_bandit", "structured_ood"}
    assert set(plan["families"]) == {"adamw", "polar", "gate_adamw", "gate_polar"}
    for family in plan["families"]:
        for scenario in names:
            configs = evaluation.family_grid(plan, family, scenario)
            if family.startswith("gate_"):
                assert len(configs) == 13 * 4 * 2 and all(c["weight_decay"] > 0 and "gate_beta" in c for c in configs)
            else:
                assert len(configs) == 13 * 5 and all("gate_beta" not in c for c in configs)
            assert len({c["beta1"] for c in configs}) == 1
