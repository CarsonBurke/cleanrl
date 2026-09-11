"""Independent numerical contracts. Run CUDA execution and compilation through mlq."""

import pytest
import torch
import torch.nn.functional as F

from cleanrl.plasticity import optimizer_proxy_model_v4 as proxy

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def full_precision():
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def problem(candidates=2, dtype=torch.float64):
    gen = torch.Generator(device="cuda").manual_seed(41)
    weights = [0.4 * torch.randn(candidates, out_dim, in_dim + 1, generator=gen,
                               device="cuda", dtype=dtype)
               for in_dim, out_dim in ((3, 5), (5, 4), (4, 1))]
    x = torch.randn(6, 3, generator=gen, device="cuda", dtype=dtype)
    target = torch.tensor([0.8, -0.4, 1.1, -0.9, 0.3, -0.2], device="cuda", dtype=dtype)
    return weights, x, target


def oracle_forward(weights, x):
    # Separate candidate networks and ordinary linear layers, not the batched VJP.
    predictions = []
    for k in range(weights[0].shape[0]):
        hidden = x
        for index, weight in enumerate(weights):
            hidden = F.linear(hidden, weight[k, :, :-1], weight[k, :, -1])
            if index != len(weights) - 1:
                hidden = hidden.tanh()
        predictions.append(hidden.squeeze(-1))
    return torch.stack(predictions)


def oracle_loss(prediction, target, actions, old_logprob, advantages, objective):
    if objective == "regression":
        return 0.5 * (prediction - target).square().mean(-1)
    logprob = torch.distributions.Normal(prediction, 0.5).log_prob(actions)
    ratio = (logprob - old_logprob).exp()
    return torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2)).mean(-1)


def frozen_batch(weights, x, target, objective):
    if objective == "regression":
        empty = target.new_empty(0)
        return target, empty, empty, empty, objective
    prediction = oracle_forward(weights, x).detach()
    actions = prediction + prediction.new_tensor([1.0, -0.8, 0.9, -1.1, 0.7, -0.6])
    ratios = prediction.new_tensor([0.65, 0.65, 1.45, 1.45, 0.95, 1.05])
    old_logprob = torch.distributions.Normal(prediction, 0.5).log_prob(actions) - ratios.log()
    advantages = prediction.new_tensor([1.2, -0.7, 0.8, -1.4, 0.5, -0.9]).expand_as(prediction)
    return target, actions, old_logprob, advantages, objective


def oracle_parts(weights, previous, x, batch):
    current = [w.detach().clone().requires_grad_() for w in weights]
    old = [w.detach().clone().requires_grad_() for w in previous]
    prediction, old_prediction = oracle_forward(current, x), oracle_forward(old, x)
    score = torch.autograd.grad(oracle_loss(prediction, *batch).sum(), prediction, retain_graph=True)[0]
    old_score = torch.autograd.grad(oracle_loss(old_prediction, *batch).sum(), old_prediction,
                                    retain_graph=True)[0]
    g = torch.autograd.grad(prediction, current, score, retain_graph=True)
    old_g = torch.autograd.grad(old_prediction, old, old_score)
    predictive = torch.autograd.grad(prediction, current, score - old_score, retain_graph=True)
    current_j_old_score = torch.autograd.grad(prediction, current, old_score)
    full = [a - b for a, b in zip(g, old_g)]
    omitted = [a - b for a, b in zip(current_j_old_score, old_g)]
    return list(g), list(predictive), full, omitted


def assert_tensors(actual, expected, *, rtol=1e-10, atol=1e-11):
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected):
        torch.testing.assert_close(observed.double(), wanted.double(), rtol=rtol, atol=atol)


def test_manual_vjp_matches_independent_three_layer_autograd():
    weights, x, _ = problem()
    leaves = [w.clone().requires_grad_() for w in weights]
    prediction = oracle_forward(leaves, x)
    score = prediction.new_tensor([[0.4, -0.8, 0.2, 1.1, -0.3, 0.7],
                                   [-0.5, 0.6, 0.9, -0.2, 0.8, -1.0]]) / x.shape[0]
    expected = torch.autograd.grad((prediction * score).sum(), leaves)
    torch.testing.assert_close(proxy.forward(weights, x), prediction, rtol=1e-11, atol=1e-12)
    assert_tensors(proxy.backward(weights, x, score), expected)


def test_gaussian_signed_clipping_scores_follow_frozen_branch_changes():
    previous = torch.zeros(2, 6, device="cuda", dtype=torch.float64)
    actions = previous.new_tensor([[1.0], [-1.0]]).expand_as(previous)
    old_ratios = previous.new_tensor([0.65, 0.65, 1.45, 1.45, 0.95, 1.05])
    new_ratios = previous.new_tensor([1.4, 1.05, 0.95, 0.65, 1.35, 0.7])
    old_logprob = torch.distributions.Normal(previous, 0.5).log_prob(actions) - old_ratios.log()
    current = actions - actions.sign() * (1.0 - 0.5 * (new_ratios / old_ratios).log()).sqrt()
    advantages = previous.new_tensor([1.2, -0.7, 0.8, -1.4, 0.5, -0.9]).expand_as(previous)
    batch = previous.new_empty(0), actions, old_logprob, advantages, "ppo"
    for prediction, inactive in ((previous, [False, True, True, False, False, False]),
                                 (current, [True, False, False, True, True, True])):
        leaf = prediction.clone().requires_grad_()
        loss = oracle_loss(leaf, *batch)
        expected_score = torch.autograd.grad(loss.sum(), leaf)[0]
        actual_score = proxy.output_score(prediction, *batch)
        torch.testing.assert_close(proxy.output_loss(prediction, *batch), loss, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(actual_score, expected_score, rtol=1e-11, atol=1e-12)
        mask = torch.tensor(inactive, device="cuda").expand_as(prediction)
        assert torch.equal(actual_score == 0, mask)


@pytest.mark.parametrize("objective", ["regression", "ppo"])
def test_predictive_full_decomposition_includes_omitted_jacobian_term(objective):
    previous, x, target = problem()
    current = [w * 1.17 + 0.08 for w in previous]
    current[-1][..., -1].add_(0.35)
    batch = frozen_batch(previous, x, target, objective)
    expected_g, predictive, full, omitted = oracle_parts(current, previous, x, batch)
    for method in proxy.METHODS:
        g, correction = proxy.gradients(current, previous, x, *batch, method)
        wanted = ([torch.zeros_like(w) for w in current] if method == "adamw"
                  else predictive if method == "predictive" else full)
        assert_tensors(g, expected_g)
        assert_tensors(correction, wanted)
    assert_tensors(full, [p + residual for p, residual in zip(predictive, omitted)])
    # The omitted term must matter to a consumer, not merely satisfy a zero identity.
    assert torch.cat([term.flatten() for term in omitted]).norm().item() > 1e-3
    if objective == "regression":
        prediction = oracle_forward(current, x)
        torch.testing.assert_close(proxy.output_score(prediction, *batch),
                                   (prediction - target) / x.shape[0], rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(proxy.output_loss(prediction, *batch), oracle_loss(prediction, *batch))


def test_zero_correction_matches_adamw_with_bias_decay_excluded():
    configs = [(beta1, beta2) for beta1 in (0.0, 0.9, 0.999) for beta2 in (0.95, 0.999)]
    initial, x, target = problem(len(configs))
    hyper = [initial[0].new_full((len(configs), 1, 1), value) for value in (0.007, 0.0, 0.0, 0.13)]
    lr, beta1, beta2, decay = hyper
    beta1[:, 0, 0] = beta1.new_tensor([config[0] for config in configs])
    beta2[:, 0, 0] = beta2.new_tensor([config[1] for config in configs])
    matrices, biases, optimizers = [], [], []
    for k, betas in enumerate(configs):
        matrices.append([torch.nn.Parameter(w[k, :, :-1].clone()) for w in initial])
        biases.append([torch.nn.Parameter(w[k, :, -1].clone()) for w in initial])
        optimizers.append(torch.optim.AdamW([
            {"params": matrices[-1], "weight_decay": 0.13},
            {"params": biases[-1], "weight_decay": 0.0},
        ], lr=0.007, betas=betas, eps=1e-8, foreach=False, fused=False))
    states = {method: ([w.clone() for w in initial], [torch.zeros_like(w) for w in initial],
                       [torch.zeros_like(w) for w in initial], torch.zeros((), device="cuda", dtype=torch.int64))
              for method in proxy.METHODS}
    zero = [torch.zeros_like(w) for w in initial]
    for iteration in range(4):
        for k, optimizer in enumerate(optimizers):
            optimizer.zero_grad(set_to_none=True)
            hidden = x + 0.07 * iteration
            for index, (matrix, bias) in enumerate(zip(matrices[k], biases[k])):
                hidden = F.linear(hidden, matrix, bias)
                if index < 2:
                    hidden = hidden.tanh()
            (0.5 * (hidden.squeeze(-1) - target.roll(iteration)).square().mean()).backward()
        gradients = [torch.stack([torch.cat((matrices[k][layer].grad, biases[k][layer].grad[:, None]), -1)
                                  for k in range(len(configs))]) for layer in range(3)]
        for optimizer in optimizers:
            optimizer.step()
        expected = [torch.stack([torch.cat((matrices[k][layer], biases[k][layer][:, None]), -1)
                                 for k in range(len(configs))]) for layer in range(3)]
        for method, (weights, m, v, step) in states.items():
            states[method] = proxy.transition(weights, initial, m, v, step, gradients, zero, *hyper, method)
            assert_tensors(states[method][0], expected)


def oracle_transition(weights, m, v, step, gradients, corrections, hyper, method):
    """Scalar candidate oracle using geometric sums rather than production bias-mass arithmetic."""
    out_w, out_m, out_v = [], [], []
    for weight, first, second, gradient, correction in zip(weights, m, v, gradients, corrections):
        layer_w, layer_m, layer_v = [], [], []
        for k in range(weight.shape[0]):
            lr, b1, b2, decay = [value[k].item() for value in hyper]
            past_mass = (1 - b1) * sum(b1 ** j for j in range(step))
            mass1 = (1 - b1) * sum(b1 ** j for j in range(step + 1))
            mass2 = (1 - b2) * sum(b2 ** j for j in range(step + 1))
            gamma = {"adamw": 0.0, "predictive": 1.0, "full": 1.0, "mars_01": 0.1, "mars_1": 1.0}[method]
            h = gradient[k] + gamma * b1 * past_mass / (1 - b1) * correction[k]
            next_m = b1 * first[k] + (1 - b1) * h
            variance_gradient = h if method.startswith("mars_") else gradient[k]
            next_v = b2 * second[k] + (1 - b2) * variance_gradient.square()
            next_w = weight[k].clone()
            next_w[:, :-1] *= 1 - lr * decay
            next_w -= lr * (next_m / mass1) / ((next_v / mass2).sqrt() + 1e-8)
            layer_w.append(next_w)
            layer_m.append(next_m)
            layer_v.append(next_v)
        out_w.append(torch.stack(layer_w))
        out_m.append(torch.stack(layer_m))
        out_v.append(torch.stack(layer_v))
    return out_w, out_m, out_v


def test_gamma_one_tracks_corrected_gradient_variance_not_raw_gradient():
    weights = [torch.tensor([[[0.7, -0.2]]], device="cuda", dtype=torch.float64)]
    m, v = [torch.full_like(weights[0], 0.1)], [torch.full_like(weights[0], 0.05)]
    g, correction = [torch.full_like(weights[0], 0.2)], [torch.full_like(weights[0], 0.4)]
    hyper = [weights[0].new_full((1, 1, 1), value) for value in (0.02, 0.9, 0.95, 0.1)]
    step = torch.tensor(3, device="cuda", dtype=torch.int64)
    full = proxy.transition(weights, weights, m, v, step, g, correction, *hyper, "full")
    mars = proxy.transition(weights, weights, m, v, step, g, correction, *hyper, "mars_1")
    for actual, method in ((full, "full"), (mars, "mars_1")):
        expected = oracle_transition(weights, m, v, 3, g, correction, hyper, method)
        for observed, wanted in zip(actual[:3], expected):
            assert_tensors(observed, wanted)
    assert_tensors(full[1], mars[1])
    assert (mars[2][0] - full[2][0]).abs().min().item() > 0.05
    assert (mars[0][0] - full[0][0]).abs().min().item() > 1e-3


@pytest.mark.parametrize("method", proxy.METHODS)
def test_compiled_fp32_multistep_startup_mass_and_previous_commit_ownership(method):
    weights, x, target = problem(3, torch.float32)
    # Deliberately unequal at startup: correction must have zero first-step mass.
    previous = [w * 0.9 - 0.07 for w in weights]
    m, v = [torch.zeros_like(w) for w in weights], [torch.zeros_like(w) for w in weights]
    step = torch.zeros((), device="cuda", dtype=torch.int64)
    hyper = [weights[0].new_tensor(values).view(3, 1, 1) for values in
             ([0.005, 0.008, 0.012], [0.0, 0.9, 0.999], [0.95, 0.999, 0.95], [0.0, 0.03, 0.13])]
    objective = "ppo" if method in ("predictive", "full") else "regression"
    batch = frozen_batch(previous, x, target, objective)
    oracle_w, oracle_previous = [w.double() for w in weights], [w.double() for w in previous]
    oracle_m, oracle_v = [w.double() for w in m], [w.double() for w in v]
    oracle_hyper = [value.double() for value in hyper]
    oracle_batch = tuple(value.double() if isinstance(value, torch.Tensor) else value for value in batch)

    def update(weights, previous, m, v, step, x, target, actions, old_logprob, advantages, lr, b1, b2, decay):
        g, correction = proxy.gradients(weights, previous, x, target, actions, old_logprob, advantages,
                                        objective, method)
        return proxy.transition(weights, previous, m, v, step, g, correction, lr, b1, b2, decay, method)

    compiled = torch.compile(update, fullgraph=True)
    for iteration in range(6):
        current_x = x + iteration * 0.025
        g, predictive, full, _ = oracle_parts(oracle_w, oracle_previous, current_x.double(), oracle_batch)
        correction = ([torch.zeros_like(w) for w in oracle_w] if method == "adamw"
                      else predictive if method == "predictive" else full)
        expected = oracle_transition(oracle_w, oracle_m, oracle_v, iteration, g, correction, oracle_hyper, method)
        # Check cancellation-sensitive moment coordinates from the exact current
        # FP32 state, separately from the independently evolved FP64 trajectory.
        # Otherwise prior parameter rounding becomes amplified into a near-zero
        # moment comparison and is incorrectly attributed to this transition.
        local_g, local_predictive, local_full, _ = oracle_parts(
            [w.double() for w in weights], [w.double() for w in previous],
            current_x.double(), oracle_batch)
        local_c = ([torch.zeros_like(w) for w in local_g] if method == "adamw"
                   else local_predictive if method == "predictive" else local_full)
        local = oracle_transition(
            [w.double() for w in weights], [w.double() for w in m], [w.double() for w in v],
            iteration, local_g, local_c, oracle_hyper, method)
        inputs = weights + previous + m + v + [step]
        snapshots = [value.clone() for value in inputs]
        next_weights, next_m, next_v, next_step = compiled(
            weights, previous, m, v, step, current_x, *batch[:-1], *hyper)
        # A pure transition must not silently commit current or previous state.
        assert_tensors(inputs, snapshots, rtol=0, atol=0)
        assert_tensors(next_weights, expected[0], rtol=3e-4, atol=3e-6)
        assert_tensors(next_m + next_v, local[1] + local[2], rtol=3e-4, atol=2e-8)
        assert next_step.item() == iteration + 1
        # Preserve caller-owned buffers, just as the compiled production boundary does.
        for old, current, updated in zip(previous, weights, next_weights):
            old.copy_(current)
            current.copy_(updated)
        for first, second, updated_first, updated_second in zip(m, v, next_m, next_v):
            first.copy_(updated_first)
            second.copy_(updated_second)
        step.copy_(next_step)
        oracle_previous, (oracle_w, oracle_m, oracle_v) = oracle_w, expected
        assert_tensors(previous, oracle_previous, rtol=3e-4, atol=3e-6)
        torch.testing.assert_close(proxy.forward(weights, current_x).double(), oracle_forward(oracle_w, current_x.double()),
                                   rtol=3e-4, atol=3e-6)
