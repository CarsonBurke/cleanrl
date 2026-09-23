"""Prospective-consolidation contracts: fast-tier identity, whitening direction, controls, capture parity."""

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_model_v8 as old
from cleanrl.plasticity import optimizer_proxy_model_v9 as proxy

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def state(dtype=torch.float64, k=3):
    gen = torch.Generator(device="cuda").manual_seed(101)
    weights = [torch.randn(k, o, i + 1, generator=gen, device="cuda", dtype=dtype) * .1
               for o, i in [(4, 3), (4, 4), (1, 4)]]
    m = [torch.randn(w.shape, generator=gen, device="cuda", dtype=dtype) * .02 for w in weights]
    v = [torch.rand(w.shape, generator=gen, device="cuda", dtype=dtype) * .1 for w in weights]
    hyper = {key: weights[0].new_tensor(values[:k]).reshape(k, 1, 1) for key, values in (
        ("lr", [.001, .003, .007]), ("beta1", [.0, .9, .9999]), ("beta2", [.95, .999, .9]),
        ("weight_decay", [.0, .01, .1]), ("head_lr_scale", [1., 1., 1.]),
        ("tier_eta", [.3, 1., .3]), ("tier_gamma", [.9, .98, .9]), ("tier_kmin", [.03, .03, .03]),
        ("tier_k0", [1., 1., 1.]), ("tier_vgamma", [.9, .9, .9]))}
    return weights, [w.clone() for w in weights], m, v, hyper


def grads_like(weights, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = [torch.randn(w.shape, generator=gen, device="cuda", dtype=w.dtype) * .2 for w in weights]
    return g, [torch.zeros_like(x) for x in g]


def run(method, hyper, steps, block, dtype=torch.float64, transition=None):
    """Roll `steps` transitions; returns final weights and the trajectory of deployed weights."""
    w, prev, m, v, _ = state(dtype)
    aux = proxy.initial_aux(w)
    for a in aux:
        a[:, 1] = hyper["tier_k0"]
    step = torch.tensor(0, device="cuda", dtype=torch.int64)
    transition = transition or proxy.transition
    for t in range(steps):
        g, c = grads_like(w, 1000 + t)
        prev = [x.clone() for x in w]
        w, m, v, aux, step = transition(w, prev, m, v, aux, step, g, c, hyper, method, 8, block)
    return w, m, v, aux, step


@pytest.mark.parametrize("method,inner", [("tier_adamw", "adamw"), ("tier_polar", "polar"),
                                          ("look_adamw", "adamw"), ("scalar_adamw", "adamw"),
                                          ("tier_vel_adamw", "adamw")])
def test_unit_gain_without_adaptation_reproduces_fast_tier_exactly(method, inner):
    _, _, _, _, hyper = state()
    identity = {**hyper, "tier_eta": torch.zeros_like(hyper["tier_eta"]), "tier_k0": torch.ones_like(hyper["tier_k0"])}
    w, m, v, aux, step = run(method, identity, 9, 4)
    w0, prev0, m0, v0, _ = state()
    step0 = torch.tensor(0, device="cuda", dtype=torch.int64)
    for t in range(9):
        g, c = grads_like(w0, 1000 + t)
        prev0 = [x.clone() for x in w0]
        w0, m0, v0, step0 = old.transition(w0, prev0, m0, v0, step0, g, c, hyper["lr"], hyper["beta1"],
                                           hyper["beta2"], hyper["weight_decay"], inner, 8, hyper["head_lr_scale"])
    for a, e in zip(w, w0):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for a, e in zip(m, m0):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    # At a boundary (step 8) the deployed weights equal phi exactly.
    assert int(step) == 9
    for a, layer in zip(aux, w):
        assert torch.equal(a[:, 1], torch.ones_like(layer))


@pytest.mark.parametrize("method", ["adamw", "polar", "adam_rms"])
def test_non_tier_families_pass_aux_through_and_match_v8(method):
    w, prev, m, v, hyper = state()
    aux = proxy.initial_aux(w)
    g, c = grads_like(w, 7)
    step = torch.tensor(5, device="cuda", dtype=torch.int64)
    got = proxy.transition(w, prev, m, v, aux, step, g, c, hyper, method, 8, 8)
    want = old.transition(w, prev, m, v, step, g, c, hyper["lr"], hyper["beta1"], hyper["beta2"],
                          hyper["weight_decay"], method, 8, hyper["head_lr_scale"])
    for a, e in zip(got[0], want[0]):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for a, e in zip(got[3], aux):
        assert a is e


def consolidate_sequence(displacements, eta=1.0, gamma=0.9, kmin=0.03, mode="param"):
    """Drive consolidate() with prescribed block displacements; return gain trajectory."""
    dtype = torch.float64
    phi = torch.zeros(1, 2, 3, device="cuda", dtype=dtype)
    aux = [torch.stack((phi, torch.ones_like(phi), *[torch.zeros_like(phi)] * 4), dim=1)]
    one = phi.new_ones(1, 1, 1)
    gains = []
    for d in displacements:
        z = [aux[0][:, 0] + d]
        z, aux = proxy.consolidate(z, aux, phi.new_tensor(1.0), eta * one, gamma * one, kmin * one, 0.9 * one,
                                   mode, False)
        gains.append(aux[0][:, 1].clone())
    return gains, aux


def test_persistent_displacements_keep_gain_high_and_alternating_shrink_it():
    persistent = [torch.full((1, 2, 3), .01, device="cuda", dtype=torch.float64)] * 30
    alternating = [torch.full((1, 2, 3), .01 * (-1) ** t, device="cuda", dtype=torch.float64) for t in range(30)]
    high, _ = consolidate_sequence(persistent)
    low, _ = consolidate_sequence(alternating)
    assert torch.equal(high[-1], torch.ones_like(high[-1]))
    assert torch.equal(low[-1], torch.full_like(low[-1], .03))
    # Monotone approach: every alternating step never raises the gain.
    for before, after in zip(low, low[1:]):
        assert bool((after <= before).all())


def test_whitening_uses_lag_one_covariance_ratio_and_exact_state_algebra():
    d0 = torch.tensor([[[.02, -.01, .0], [.005, .005, .005]]], device="cuda", dtype=torch.float64)
    d1 = torch.tensor([[[.01, .01, .0], [.005, -.005, .0]]], device="cuda", dtype=torch.float64)
    gains, aux = consolidate_sequence([d0, d1], eta=.5, gamma=.8)
    nu = .2 * d0.square() * .8 + .2 * d1.square()
    c = .2 * d1 * d0
    expected = torch.clamp(gains[0] * torch.exp(.5 * c / (nu + 1e-30)), max=1.0).clamp(min=.03)
    torch.testing.assert_close(gains[1], expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(aux[0][:, 3], nu, rtol=1e-12, atol=0)
    torch.testing.assert_close(aux[0][:, 2], c, rtol=1e-12, atol=0)
    torch.testing.assert_close(aux[0][:, 4], d1, rtol=1e-12, atol=1e-18)
    # phi advanced by K d at each boundary, from zero.
    phi = gains[0] * d0 + gains[1] * d1
    torch.testing.assert_close(aux[0][:, 0], phi, rtol=1e-12, atol=1e-15)


def test_scalar_control_shares_one_gain_per_candidate():
    gen = torch.Generator(device="cuda").manual_seed(3)
    seq = [torch.randn(1, 2, 3, generator=gen, device="cuda", dtype=torch.float64) * .01 for _ in range(12)]
    gains, _ = consolidate_sequence(seq, mode="scalar")
    for g in gains:
        assert torch.equal(g, g.flatten()[0].expand_as(g))
    param_gains, _ = consolidate_sequence(seq, mode="param")
    assert not torch.equal(param_gains[-1], param_gains[-1].flatten()[0].expand_as(param_gains[-1]))


def test_off_boundary_steps_leave_consolidation_state_untouched():
    w, prev, m, v, hyper = state()
    aux = proxy.initial_aux(w)
    g, c = grads_like(w, 11)
    step = torch.tensor(5, device="cuda", dtype=torch.int64)
    got = proxy.transition(w, prev, m, v, aux, step, g, c, hyper, "tier_adamw", 8, 8)
    for a, e in zip(got[3], aux):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    want = old.transition(w, prev, m, v, step, g, c, hyper["lr"], hyper["beta1"], hyper["beta2"],
                          hyper["weight_decay"], "adamw", 8, hyper["head_lr_scale"])
    for a, e in zip(got[0], want[0]):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["tier_polar", "tier_vel_adamw", "scalar_adamw"])
def test_compiled_matches_eager_through_a_boundary(method):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    _, _, _, _, hyper = state(torch.float32)
    compiled = torch.compile(proxy.transition, fullgraph=True, options={"triton.cudagraphs": False})
    eager = run(method, hyper, 8, 4, torch.float32)
    fast = run(method, hyper, 8, 4, torch.float32, transition=compiled)
    for a, e in zip(fast[0], eager[0]):
        torch.testing.assert_close(a, e, rtol=3e-4, atol=3e-6)
    for a, e in zip(fast[3], eager[3]):
        torch.testing.assert_close(a, e, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("method,scenario", [
    ("tier_polar", {"name": "clipped_bandit", "objective": "ppo", "samples": 4096, "batch_size": 256, "epochs": 8, "drift": True, "block": 8}),
    ("tier_vel_adamw", {"name": "iid_online", "objective": "regression", "samples": 512, "batch_size": 1, "epochs": 1, "drift": False, "block": 32}),
    ("scalar_adamw", {"name": "drifting_reuse", "objective": "regression", "samples": 4096, "batch_size": 64, "epochs": 8, "drift": True, "block": 8}),
])
def test_learner_capture_audit_passes_and_replays_through_boundaries(method, scenario):
    """Real RoundLearner: compile, conditioned audit, graph capture, then replay past a block boundary."""
    from types import SimpleNamespace

    from cleanrl.plasticity import network_bayes_stream_v2 as reference
    from cleanrl.plasticity import optimizer_proxy_eval_v9 as protocol

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.compiler.reset()
    args = SimpleNamespace(hidden=64, input_dim=17)
    initial = reference.init_weights(args, protocol.generator(1, 1), "cuda")
    grid = [{"lr": 1e-3, "beta1": .9, "beta2": .99, "weight_decay": .1, "head_lr_scale": 1.0,
             "tier_eta": 1.0, "tier_gamma": .9, "tier_kmin": .03, "tier_k0": 1.0, "tier_vgamma": .9},
            {"lr": 3e-3, "beta1": .5, "beta2": .999, "weight_decay": .0, "head_lr_scale": .5,
             "tier_eta": .3, "tier_gamma": .98, "tier_kmin": .03, "tier_k0": 1.0, "tier_vgamma": .9}]
    learner = protocol.RoundLearner(initial, grid, scenario, method)
    gen = torch.Generator(device="cuda").manual_seed(9)
    b = scenario["batch_size"]
    learner.x.copy_(torch.randn(b, 17, generator=gen, device="cuda"))
    learner.target.copy_(torch.randn(b, generator=gen, device="cuda"))
    learner.action_noise.copy_(torch.randn(b, generator=gen, device="cuda"))
    learner.reward_noise.copy_(torch.randn(b, generator=gen, device="cuda"))
    graph = learner.capture()
    rounds = 3 * scenario["block"] // scenario["epochs"]
    for _ in range(rounds):
        learner.x.copy_(torch.randn(b, 17, generator=gen, device="cuda"))
        learner.target.copy_(torch.randn(b, generator=gen, device="cuda"))
        graph.replay()
    torch.cuda.synchronize()
    assert bool(learner.valid.all())
    assert int(learner.step) == rounds * scenario["epochs"]
    gain_mean, gain_std = learner.gain_summary()
    assert torch.isfinite(gain_mean).all() and bool((gain_mean <= 1).all())
    if method == "scalar_adamw":
        assert bool((gain_std == 0).all())
    else:
        assert bool((gain_std > 0).all()), "adaptive gains should disperse after boundaries"
    # After a boundary the deployed weights equal the consolidated tier.
    for w, a in zip(learner.weights, learner.aux):
        torch.testing.assert_close(w, a[:, 0], rtol=0, atol=0)
