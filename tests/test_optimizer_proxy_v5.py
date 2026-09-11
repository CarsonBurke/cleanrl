"""Fresh-observation clocks and within-reuse gradient-estimator invariants."""

import math

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_model_v4 as base
from cleanrl.plasticity import optimizer_proxy_model_v5 as proxy

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def tensors():
    w = [torch.tensor([[[0.3, -0.7, 0.1]], [[-0.2, 0.5, 0.8]], [[0.6, 0.1, -0.4]]],
                      device="cuda", dtype=torch.float64)]
    m, v = [torch.zeros_like(w[0])], [torch.zeros_like(w[0])]
    previous = [w[0].clone()]
    hyper = [w[0].new_tensor(x).reshape(3, 1, 1) for x in
             ([.01, .02, .005], [0., .9, .99999], [.95, .999, .9], [.0, .1, .03])]
    return w, previous, m, v, torch.zeros((), device="cuda", dtype=torch.int64), hyper


def commit(w, previous, m, v, step, output):
    nw, nm, nv, ns = output
    for p, old, new, first, second, nfirst, nsecond in zip(previous, w, nw, m, v, nm, nv):
        p.copy_(old)
        old.copy_(new)
        first.copy_(nfirst)
        second.copy_(nsecond)
    step.copy_(ns)


@pytest.mark.parametrize("method", ["adamw", "predictive", "full"])
def test_one_epoch_round_clock_equals_original_algorithm(method):
    w, previous, m, v, step, hyper = tensors()
    for t in range(7):
        g = [torch.sin(w[0] * (t + 1)) + .2]
        c = [torch.zeros_like(g[0]) if method == "adamw" else (w[0] - previous[0]) * (t + .5)]
        wanted = base.transition(w, previous, m, v, step, g, c, *hyper, method)
        actual = proxy.transition(w, previous, m, v, step, g, c, *hyper, "round_" + method, 1)
        for a, b in zip(actual[:3], wanted[:3]):
            for av, bv in zip(a, b):
                torch.testing.assert_close(av, bv, rtol=1e-12, atol=1e-13)
        commit(w, previous, m, v, step, actual)


def test_full_reuse_preserves_estimation_error_instead_of_recounting_noise():
    w, previous, m, v, step, hyper = tensors()
    lr, beta1, beta2, decay = hyper
    x = w[0].new_tensor([[1., -.3, 1.], [-.2, .8, 1.], [.4, .1, 1.], [-.7, -.2, 1.]])
    batch_error = None
    frozen_variance = None
    for t in range(12):
        group, epoch = divmod(t, 4)
        y = w[0].new_tensor([.2, -.4, .7, .1]) + (group * .3)
        def gradient(weights):
            prediction = weights[0][:, 0] @ x.T
            return [((prediction - y) @ x / len(x))[:, None]]
        g, old_g = gradient(w), gradient(previous)
        c = [g[0] - old_g[0]]
        # Independent normalized-estimator recurrence; no production bias helper.
        count = group + 1
        old_mass = 1 - beta1.pow(group)
        mass = 1 - beta1.pow(count)
        if epoch == 0:
            expected_m = beta1 * m[0] + (1 - beta1) * g[0] + beta1 * old_mass * c[0]
            expected_v = beta2 * v[0] + (1 - beta2) * g[0].square()
        else:
            expected_m = m[0] + mass * c[0]
            expected_v = v[0].clone()
        expected_w = w[0].clone()
        expected_w[..., :-1] *= 1 - lr * decay
        expected_w -= lr * (expected_m / mass) / ((expected_v / (1 - beta2.pow(count))).sqrt() + 1e-8)
        output = proxy.transition(w, previous, m, v, step, g, c, *hyper, "round_full", 4)
        torch.testing.assert_close(output[0][0], expected_w, rtol=1e-10, atol=1e-12)
        q_error = output[1][0] / mass - g[0]
        if epoch == 0:
            batch_error = q_error.clone()
            frozen_variance = output[2][0].clone()
            if group:
                assert q_error[1:].norm() > 1e-3
        else:
            torch.testing.assert_close(q_error, batch_error, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(output[2][0], frozen_variance, rtol=0, atol=0)
        commit(w, previous, m, v, step, output)


def test_compiled_fresh_clock_matches_scalar_geometric_sum_oracle():
    w, previous, m, v, step, hyper = tensors()
    w, previous, m, v = [[x.float() for x in values] for values in (w, previous, m, v)]
    hyper = [x.float() for x in hyper]
    compiled = torch.compile(proxy.transition, fullgraph=True, options={"triton.cudagraphs": False})
    for t in range(10):
        g = [torch.sin(w[0] + t * .2)]
        correction = [torch.full_like(w[0], .015 * (1 if t % 2 else -1))]
        snapshots = [x.clone() for x in w + previous + m + v + [step]]
        result = compiled(w, previous, m, v, step, g, correction, *hyper, "round_predictive", 3)
        for k in range(3):
            lr, b1, b2, wd = [float(x[k]) for x in hyper]
            count = t // 3 + 1
            mass = (1-b1) * sum(b1 ** j for j in range(count))
            past = (1-b1) * sum(b1 ** j for j in range(count-1))
            variance_mass = (1-b2) * sum(b2 ** j for j in range(count))
            first = m[0][k].double()
            second = v[0][k].double()
            if t % 3 == 0:
                first = b1*first + (1-b1)*g[0][k].double() + b1*past*correction[0][k].double()
                second = b2*second + (1-b2)*g[0][k].double().square()
            else:
                first = first + mass*correction[0][k].double()
            expected = w[0][k].double().clone()
            expected[..., :-1] *= 1-lr*wd
            expected -= lr*(first/mass)/((second/variance_mass).sqrt()+1e-8)
            torch.testing.assert_close(result[0][0][k].double(), expected, rtol=3e-4, atol=3e-6)
        for actual, wanted in zip(w + previous + m + v + [step], snapshots):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        commit(w, previous, m, v, step, result)


def test_captured_round_isolates_invalid_candidate_and_preserves_replay():
    from cleanrl.plasticity.optimizer_proxy_eval_v5 import RoundLearner

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(81)
    initial = [torch.randn(o, i + 1, device="cuda", generator=gen) * .1 for o, i in [(4, 3), (4, 4), (1, 4)]]
    grid = [{"lr": .001, "beta1": .9, "beta2": .95, "weight_decay": 0.}] * 2
    learner = RoundLearner(initial, grid, {"batch_size": 8, "epochs": 3, "objective": "ppo"}, "round_full")
    learner.x.copy_(torch.randn(8, 3, device="cuda", generator=gen))
    learner.target.copy_(torch.randn(8, device="cuda", generator=gen))
    learner.action_noise.copy_(torch.randn(8, device="cuda", generator=gen))
    learner.reward_noise.copy_(torch.randn(8, device="cuda", generator=gen))
    learner.weights[0][1].fill_(float("nan"))
    before = [x.clone() for x in learner.mutable]
    graph = learner.capture()
    for actual, wanted in zip(learner.mutable, before):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0, equal_nan=True)
    graph.replay()
    assert learner.valid.tolist() == [True, False]
    assert int(learner.step) == 3
    # Repair numeric state deliberately: an already invalid trial cannot re-enter.
    for values in (learner.weights, learner.previous, learner.m, learner.v):
        for tensor in values:
            tensor[1].copy_(tensor[0])
    graph.replay()
    assert learner.valid.tolist() == [True, False]
    assert all(torch.isfinite(w[0]).all() for w in learner.weights)
