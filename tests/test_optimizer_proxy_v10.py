"""v10 contracts: look_polar equals polar at K = 1, applies the uniform Lookahead algebra, and v9 families pass through."""

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_model_v10 as proxy
from cleanrl.plasticity import optimizer_proxy_model_v8 as old
from cleanrl.plasticity import optimizer_proxy_model_v9 as prior
from tests.test_optimizer_proxy_v9 import grads_like, state

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def roll(transition, method, hyper, steps, block):
    w, prev, m, v, _ = state()
    aux = proxy.initial_aux(w)
    for a in aux:
        a[:, 1] = hyper["tier_k0"]
    step = torch.tensor(0, device="cuda", dtype=torch.int64)
    for t in range(steps):
        g, c = grads_like(w, 1000 + t)
        prev = [x.clone() for x in w]
        w, m, v, aux, step = transition(w, prev, m, v, aux, step, g, c, hyper, method, 8, block)
    return w, m, v, aux, step


def test_look_polar_at_unit_gain_is_polar_exactly():
    _, _, _, _, hyper = state()
    w, m, _, aux, step = roll(proxy.transition, "look_polar", hyper, 9, 4)
    w0, prev0, m0, v0, _ = state()
    step0 = torch.tensor(0, device="cuda", dtype=torch.int64)
    for t in range(9):
        g, c = grads_like(w0, 1000 + t)
        prev0 = [x.clone() for x in w0]
        w0, m0, v0, step0 = old.transition(w0, prev0, m0, v0, step0, g, c, hyper["lr"], hyper["beta1"],
                                           hyper["beta2"], hyper["weight_decay"], "polar", 8, hyper["head_lr_scale"])
    for a, e in zip(w + m, w0 + m0):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert all(torch.equal(a[:, 1], torch.ones_like(a[:, 1])) for a in aux)


def test_look_polar_consolidates_uniformly_at_the_boundary_and_holds_phi_between():
    _, _, _, _, hyper = state()
    k0 = hyper["tier_k0"].new_tensor([.1, .25, .5]).reshape(3, 1, 1)
    hyper = {**hyper, "tier_k0": k0}
    w3, _, _, aux3, _ = roll(proxy.transition, "look_polar", hyper, 3, 4)
    w4, _, _, aux4, step = roll(proxy.transition, "look_polar", hyper, 4, 4)
    _, phi0, _, _, _ = state()
    assert int(step) == 4
    for layer3, layer4, a3, a4, phi in zip(w3, w4, aux3, aux4, phi0):
        assert torch.equal(a3[:, 0], phi)  # phi untouched off the boundary; the transient tier moved
        assert not torch.equal(layer3, phi)
        # One more polar step from the transient, then phi' = phi + K (z - phi); weights reset to phi'.
        assert torch.equal(a4[:, 0], layer4)
        assert torch.equal(a4[:, 1], k0.expand_as(a4[:, 1]))
    # Gain never adapts for the fixed mode even with eta > 0.
    assert all(torch.equal(a[:, 1], k0.expand_as(a[:, 1])) for a in aux4)


@pytest.mark.parametrize("method", ["adamw", "polar", "tier_adamw", "look_adamw", "scalar_adamw", "tier_vel_adamw", "tier_polar"])
def test_every_v9_family_is_unchanged(method):
    _, _, _, _, hyper = state()
    new = roll(proxy.transition, method, hyper, 9, 4)
    ref = roll(prior.transition, method, hyper, 9, 4)
    for a, e in zip(new[0] + new[1] + new[2] + new[3], ref[0] + ref[1] + ref[2] + ref[3]):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_learner_capture_audit_passes_and_replays_for_look_polar():
    from types import SimpleNamespace

    from cleanrl.plasticity import network_bayes_stream_v2 as reference
    from cleanrl.plasticity import optimizer_proxy_eval_v10 as protocol

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.compiler.reset()
    scenario = {"name": "iid_online", "objective": "regression", "samples": 512, "batch_size": 1, "epochs": 1, "drift": False, "block": 32}
    args = SimpleNamespace(hidden=64, input_dim=17)
    initial = reference.init_weights(args, protocol.generator(1, 1), "cuda")
    grid = [{"lr": 1e-3, "beta1": .995, "beta2": .95, "weight_decay": 0.0, "head_lr_scale": 8.0,
             "tier_eta": 0.0, "tier_gamma": .9, "tier_kmin": .03, "tier_k0": k, "tier_vgamma": .9} for k in (.1, .25, .5)]
    learner = protocol.RoundLearner(initial, grid, scenario, "look_polar")
    gen = torch.Generator(device="cuda").manual_seed(9)
    learner.x.copy_(torch.randn(1, 17, generator=gen, device="cuda"))
    learner.target.copy_(torch.randn(1, generator=gen, device="cuda"))
    graph = learner.capture()
    for _ in range(3 * scenario["block"]):
        learner.x.copy_(torch.randn(1, 17, generator=gen, device="cuda"))
        learner.target.copy_(torch.randn(1, generator=gen, device="cuda"))
        graph.replay()
    mean, std = learner.gain_summary()
    assert torch.allclose(mean.cpu(), torch.tensor([.1, .25, .5]))
    assert torch.equal(std, torch.zeros_like(std))
    assert bool(learner.valid.all())
