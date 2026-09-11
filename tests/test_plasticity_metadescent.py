"""Direct hypergradient and causal gate contracts; CUDA execution through mlq."""

import pytest
import torch

from cleanrl.plasticity import unit_metadescent_stream_v1 as meta
from cleanrl.plasticity.unit_bayes_stream_v1 import forward, init_weights
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def make_learner(method, meta_lr=0.002):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = meta.Args(input_dim=2, hidden=3, context_rank=2, graph_steps=4)
    gen = torch.Generator(device="cuda").manual_seed(1)
    initial = init_weights(a, gen, "cuda")
    projections = [torch.randn(w.shape[-1] - 1, a.context_rank, generator=gen, device="cuda")
                   for w in initial]
    xs = torch.randn(8, a.input_dim, generator=gen, device="cuda")
    ys = torch.randn(8, generator=gen, device="cuda")
    shifts = torch.ones(8, 2, device="cuda", dtype=torch.long)
    return meta.Learner(method, [(0.001, meta_lr)], initial, projections, a, xs, ys, shifts)


@pytest.mark.parametrize("method", ["unit", "bias", "shared", "shuffle"])
def test_delayed_meta_update_follows_true_direct_next_loss_derivative(method):
    learner = make_learner(method)
    learner.theta.copy_(torch.linspace(-0.1, 0.2, learner.theta.numel(), device="cuda").reshape_as(learner.theta))
    before = [w.clone() for w in learner.weights]
    theta0 = learner.theta.clone()
    learner.update()
    # No prior observation: no meta-learning on this example's own error.
    torch.testing.assert_close(learner.theta, theta0, rtol=0, atol=0)
    theta = theta0.clone().requires_grad_()
    gate, _ = meta.gate_value_and_log_derivative(
        (theta * learner.previous_features).sum(-1) + meta.IDENTITY_LOGIT)
    if method == "shuffle":
        # The inverse stored by production maps source -> destination. Sort it
        # to reconstruct destination -> source for an independent forward model.
        gate = gate.index_select(-1, learner.previous_inverse.argsort())
    else:
        gate = gate.expand(-1, learner.units)
    transformed = []
    start = 0
    for w, applied in zip(before, learner.previous_step):
        stop = start + w.shape[1]
        destination_gate = gate[:, start:stop, None]
        # Hold Adam's direction fixed, and differentiate ONLY the previous gate.
        direction = applied / destination_gate.detach()
        transformed.append(w + direction * destination_gate)
        start = stop
    prediction = forward(transformed, learner.xs[1:2])[2].squeeze()
    loss = 0.5 * (prediction - learner.ys[1]).square()
    direct = torch.autograd.grad(loss, theta)[0]
    denominator = torch.where(direct.abs() > 0, direct.abs(), torch.ones_like(direct))
    expected = theta0 - learner.meta_lr * direct / denominator
    learner.update()
    torch.testing.assert_close(learner.theta, expected, rtol=1e-5, atol=1e-7)
    # The magnitude accumulator also pins the actual hypergradient, not merely
    # its sign (normalization alone would hide a wrong derivative magnitude).
    torch.testing.assert_close(learner.meta_v, (1 - learner.a.meta_beta) * direct.square(),
                               rtol=2e-4, atol=1e-15)


@torch.no_grad()
def test_current_label_cannot_choose_its_own_gate():
    left, right = make_learner("unit"), make_learner("unit")
    left.update()
    right.update()
    left.ys[1] = -10
    right.ys[1] = 10
    left.update()
    right.update()
    torch.testing.assert_close(left.gate_sum, right.gate_sum, rtol=0, atol=0)
    assert not torch.equal(left.theta, right.theta)


@torch.no_grad()
def test_disabled_meta_learning_recovers_adam_without_hidden_rate_change():
    baseline, contextual = make_learner("adam", 0), make_learner("unit", 0)
    for _ in range(8):
        baseline.update()
        contextual.update()
    for plain, gated in zip(baseline.weights, contextual.weights):
        torch.testing.assert_close(plain, gated, rtol=0, atol=0)
