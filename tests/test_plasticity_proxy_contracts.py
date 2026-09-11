"""Observable mathematical and replay contracts; run CUDA tests through mlq."""

from types import SimpleNamespace

import pytest
import torch

from cleanrl.plasticity import sample_stream as sample
from cleanrl.plasticity import unit_bayes_stream_v1 as bayes
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_neuron_jacobian_matches_autograd():
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = bayes.Args(input_dim=3, hidden=5)
    gen = torch.Generator(device="cuda").manual_seed(1)
    weights = [w.unsqueeze(0).requires_grad_() for w in bayes.init_weights(a, gen, "cuda")]
    x = torch.randn(3, generator=gen, device="cuda")
    output, inputs, sensitivity = bayes.sample_state(weights, x)
    grads = torch.autograd.grad(output.sum(), weights)
    for observed, inp, j in zip(grads, inputs, sensitivity):
        torch.testing.assert_close(observed, j.unsqueeze(-1) * inp.unsqueeze(1))
    torch.testing.assert_close(output, bayes.forward(weights, x.unsqueeze(0))[2].squeeze(-1))


def test_linear_subproblem_recovers_gaussian_posterior():
    # Zero hidden activations and output weights reduce the observable model to
    # one unknown bias, whose exact Gaussian posterior is available analytically.
    a = bayes.Args(input_dim=1, hidden=1, diffusion=0, known_noise=True)
    initial = [torch.zeros(1, 2, device="cuda") for _ in range(3)]
    xs = torch.zeros(4, 1, device="cuda")
    ys = torch.tensor([2.0, -1.0, 4.0, 3.0], device="cuda")
    learner = bayes.Learner("unit", (2.0,), initial, a, xs, ys, ys, torch.ones_like(ys))
    for n in range(1, 5):
        learner.update()
        posterior_mean = ys[:n].sum() / (n + 1)
        posterior_variance = 1.0 / (n + 1)
        torch.testing.assert_close(bayes.sample_state(learner.weights, xs[0])[0][0], posterior_mean)
        torch.testing.assert_close(learner.cov[2][0, 0, -1, -1], ys.new_tensor(posterior_variance))


def test_clean_monitor_targets_do_not_change_learning():
    a = bayes.Args(input_dim=3, hidden=4, diffusion=0)
    gen = torch.Generator(device="cuda").manual_seed(1)
    initial = bayes.init_weights(a, gen, "cuda")
    xs = torch.randn(8, 3, generator=gen, device="cuda")
    ys = torch.randn(8, generator=gen, device="cuda")
    left = bayes.Learner("unit", (0.1, 1.0), initial, a, xs, ys, ys, torch.ones_like(ys))
    right = bayes.Learner("unit", (0.1, 1.0), initial, a, xs, ys, ys + 100, torch.ones_like(ys))
    for _ in range(8):
        left.update()
        right.update()
    for l, r in zip(left.weights + left.cov, right.weights + right.cov):
        torch.testing.assert_close(l, r, rtol=0, atol=0)
    assert not torch.equal(left.error, right.error)


@pytest.mark.parametrize("batch", [1, 8])
@torch.no_grad()
def test_sample_graph_preserves_causal_updates_and_replay_reset(batch):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    methods = list(sample.METHODS)
    a = SimpleNamespace(seeds=1, lr_grid=[1e-4, 1e-3, 1e-2], batch=batch, ema=0.999,
                        methods=methods, switch_at=0.5, huber_k=1.345, var_floor=0.02,
                        nu=5.0, readout_lr=0.002, readout_decay=0.0, cap=20.0)
    gen = torch.Generator(device="cuda").manual_seed(1)
    initial = sample.init_mlp(1, 3, gen, "cuda")
    xs = torch.randn(16, 1, batch, sample.D_IN, generator=gen, device="cuda")
    clean = torch.randn(16, 1, batch, 1, generator=gen, device="cuda")
    sigma = torch.rand(16, 1, batch, generator=gen, device="cuda") + 0.5
    ys = clean + sigma.unsqueeze(-1) * torch.randn(clean.shape, generator=gen, device="cuda")
    permutations = torch.rand(16, batch, generator=gen, device="cuda").argsort(-1)
    states = [sample.make_state(m, initial, 2, batch) for m in methods]
    counter = torch.zeros(1, dtype=torch.long, device="cuda")
    lr = torch.tensor(a.lr_grid, device="cuda").view(1, 3, 1, 1)
    args = (states, counter, xs, ys, clean, sigma, permutations, lr, a, 4)
    mutable = sample.tensors(states) + [counter]
    before = [t.clone() for t in mutable]
    for _ in range(8):
        sample.update(*args)
    expected = [t.clone() for t in mutable]
    for actual, original in zip(mutable, before):
        actual.copy_(original)
    compiled, single, chunk, reset, _ = sample.capture_updates(args, 4)
    for actual, original in zip(mutable, before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    for _ in range(8):
        compiled(*args)
    torch.cuda.synchronize()
    compiled_expected = [t.clone() for t in mutable]
    # Adam can amplify reduction-order roundoff near zero first moments.
    # Keep eager numerical equivalence separate from exact replay equivalence.
    for actual, reference in zip(compiled_expected, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-5)
    reset()
    chunk.replay()
    chunk.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(mutable, compiled_expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert counter.item() == 8
    reset()
    for _ in range(8):
        single.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(mutable, compiled_expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@torch.no_grad()
def test_shuffled_covariance_control_does_not_relabel_neuron_histories():
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    a = bayes.Args(input_dim=3, hidden=4, diffusion=0)
    gen = torch.Generator(device="cuda").manual_seed(1)
    initial = bayes.init_weights(a, gen, "cuda")
    xs = torch.randn(16, 3, generator=gen, device="cuda")
    ys = torch.randn(16, generator=gen, device="cuda")
    matched = bayes.Learner("unit", (10.0,), initial, a, xs, ys, ys, torch.ones_like(ys))
    shuffled = bayes.Learner("shuffle", (10.0,), initial, a, xs, ys, ys, torch.ones_like(ys))
    for _ in range(16):
        matched.update()
        shuffled.update()
    left = bayes.forward(matched.weights, xs)[2]
    right = bayes.forward(shuffled.weights, xs)[2]
    # Fixed routing plus inverse-routed conditioning previously made the whole
    # control identical to unit learning from isotropic priors. A real broken
    # correspondence changes the learned predictor, not just storage labels.
    assert (left - right).abs().max().item() > 1e-4
