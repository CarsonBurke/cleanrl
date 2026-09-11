"""Deterministic CUDA algebra and capture contracts; not stock performance runs."""

import math
from contextlib import contextmanager

import pytest
import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity.predictive_conditional_stack_v7 import (
    StateConditionedStack,
    composition_features,
)
from cleanrl.plasticity.predictive_dynamic_nig_v6 import DynamicNIG
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def exact_matmul_runtime():
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)


def stream(count=24, input_dim=14):
    time = torch.arange(count, device='cuda', dtype=torch.float32)
    coordinate = torch.arange(input_dim, device='cuda', dtype=torch.float32)
    xs = 8 * torch.sin(.37 * time[:, None] + .23 * coordinate[None, :])
    xs += 3 * torch.cos(.11 * time[:, None] * (coordinate[None, :] + 1))
    ys = 1.4 * torch.sin(.51 * time) + .7 * torch.cos(.19 * time)
    return xs, ys


def public_state(model):
    """Independent inventory makes omission from state_tensors observable."""
    base, meta, adam = model.base, model.meta, model.context_adam
    return [base.mean, base.cov, base.alpha, base.beta, base.log_all,
            base.log_static, base.log_dynamic, base.observations,
            meta.mean, meta.cov, meta.alpha, meta.beta,
            adam.weights, adam.m, adam.v, adam.steps, model.observations]


@contextmanager
def captured_update(model, x, y):
    """Warm, capture, and replay one compiled callable without changing state."""
    initial = [value.clone() for value in model.state_tensors()]

    def restore():
        for value, saved in zip(model.state_tensors(), initial, strict=True):
            value.copy_(saved)

    compiled = torch.compile(model.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(side):
            compiled(x, y)
            restore()
            compiled(x, y)
        side.synchronize()
        restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = compiled(x, y)
        restore()
        yield graph, output, restore
    finally:
        side.synchronize()
        restore()
        torch.cuda.synchronize()


def independent_design(x, prelabel_base):
    """FP64 reconstruction, independent of composition_features implementation."""
    anchor = prelabel_base[1].double()
    raw = prelabel_base[4:].double()
    main = torch.cat((anchor[None], (raw - anchor) / math.sqrt(len(raw))))
    lag_rows = x.double().reshape(-1, 7)
    context = torch.tanh(lag_rows[max(0, len(lag_rows) - 8):].mean(0)) / math.sqrt(7)
    design = torch.zeros((3, 8 * len(main)), device=x.device, dtype=torch.float64)
    design[0, 0] = anchor
    design[1, :len(main)] = main
    design[2, :len(main)] = main
    for channel in range(7):
        start = (channel + 1) * len(main)
        design[2, start:start + len(main)] = context[channel] * main
    return design


def direct_ridge(designs, residuals, prior):
    """Independent full-prefix NIG sufficient-statistic solve, not rank updates."""
    precision = torch.eye(designs.shape[1], device=designs.device, dtype=torch.float64) / prior
    precision += designs.T @ designs
    covariance = torch.linalg.inv(precision)
    information = designs.T @ residuals
    mean = torch.linalg.solve(precision, information)
    beta = 1 + .5 * (residuals @ residuals - information @ mean)
    return covariance, mean, beta


@torch.no_grad()
def test_captured_predictable_designs_match_independent_fp64_batch_ridge():
    model = StateConditionedStack(14, 'cuda')
    xs, ys = stream()
    x, y = xs[0].clone(), ys[0].clone()
    designs, residuals = [], []
    base_start = len(model.meta.output_names)
    base_stop = base_start + len(model.base.output_names)
    with captured_update(model, x, y) as (graph, output, _):
        for step in range(len(xs)):
            x.copy_(xs[step])
            y.copy_(ys[step])
            graph.replay()
            prediction = output.clone()
            base = prediction[base_start:base_stop]
            h = independent_design(x, base)
            anchor = base[1]
            # The pre-label posterior uses only designs/residuals of earlier rows.
            for family in range(3):
                for prior_index, prior in enumerate(model.priors):
                    index = family * len(model.priors) + prior_index
                    if designs:
                        history = torch.stack(designs)[:, family]
                        _, mean_before, _ = direct_ridge(history, torch.stack(residuals), prior)
                    else:
                        mean_before = torch.zeros(model.feature_dim, device='cuda', dtype=torch.float64)
                    expected = anchor.double() + h[family] @ mean_before
                    torch.testing.assert_close(prediction[index].double(), expected, rtol=2e-4, atol=2e-6)
            designs.append(h)
            residuals.append((y - anchor).double())
            history, targets = torch.stack(designs), torch.stack(residuals)
            for family in range(3):
                for prior_index, prior in enumerate(model.priors):
                    index = family * len(model.priors) + prior_index
                    covariance, mean, beta = direct_ridge(history[:, family], targets, prior)
                    torch.testing.assert_close(model.meta.cov[index].double(), covariance,
                                               rtol=2e-4, atol=2e-6)
                    torch.testing.assert_close(model.meta.mean[index].double(), mean,
                                               rtol=2e-4, atol=2e-6)
                    torch.testing.assert_close(model.meta.beta[index], beta, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(model.meta.alpha,
                                       torch.tensor(2 + .5 * (step + 1), device='cuda', dtype=torch.float64),
                                       rtol=0, atol=0)
        assert (torch.linalg.eigvalsh(model.meta.cov.double()) > 0).all()


@torch.no_grad()
def test_same_graph_current_label_independence_and_complete_future_restoration():
    model = StateConditionedStack(14, 'cuda')
    xs, ys = stream(10)
    for x_row, y_row in zip(xs[:8], ys[:8]):
        model.update(x_row, y_row)
    x, y = xs[8].clone(), torch.tensor(-4., device='cuda')
    initial = [value.clone() for value in public_state(model)]
    with captured_update(model, x, y) as (graph, output, restore):
        for value, expected in zip(public_state(model), initial, strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        graph.replay()
        low = output.clone()
        low_state = [value.clone() for value in public_state(model)]
        low_geometry = model.meta.cov.clone()
        y.zero_()
        graph.replay()
        low_future = output.clone()
        low_future_state = [value.clone() for value in public_state(model)]

        restore()
        y.fill_(4.)
        graph.replay()
        torch.testing.assert_close(output, low, rtol=0, atol=0)
        # Meta geometry cannot consult the updated base posterior: changing only
        # the current label must leave its same-step design/geometry identical.
        torch.testing.assert_close(model.meta.cov, low_geometry, rtol=0, atol=0)
        assert not torch.allclose(model.base.mean, low_state[0])
        assert not torch.allclose(model.meta.mean, low_state[8])
        assert not torch.allclose(model.context_adam.weights, low_state[12])
        y.zero_()
        graph.replay()
        t = len(model.priors)
        assert not torch.allclose(output[2 * t:3 * t], low_future[2 * t:3 * t])
        assert not torch.allclose(output[-len(model.context_adam.lrs):],
                                  low_future[-len(model.context_adam.lrs):])

        restore()
        y.fill_(-4.)
        graph.replay()
        torch.testing.assert_close(output, low, rtol=0, atol=0)
        for value, expected in zip(public_state(model), low_state, strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        y.zero_()
        graph.replay()
        torch.testing.assert_close(output, low_future, rtol=0, atol=0)
        for value, expected in zip(public_state(model), low_future_state, strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


@torch.no_grad()
def test_context_uses_recent_state_but_amplitude_and_plain_controls_do_not():
    base = DynamicNIG(70, 'cuda')
    xs, ys = stream(12, 70)
    for x, y in zip(xs[:-1], ys[:-1]):
        base.update(x, y)
    x = xs[-1]
    prelabel = base.update(x, ys[-1])
    baseline = composition_features(x, prelabel)
    changed_recent = x.clone().reshape(-1, 7)
    changed_recent[-8:] = -changed_recent[-8:]
    different = composition_features(changed_recent.flatten(), prelabel)
    torch.testing.assert_close(baseline[:2], different[:2], rtol=0, atol=0)
    g_dim = 1 + len(base.configs)
    torch.testing.assert_close(baseline[2, :g_dim], different[2, :g_dim], rtol=0, atol=0)
    assert not torch.allclose(baseline[2, g_dim:], different[2, g_dim:])
    changed_old = x.clone().reshape(-1, 7)
    changed_old[:2] += 100
    torch.testing.assert_close(composition_features(changed_old.flatten(), prelabel), baseline,
                               rtol=0, atol=0)
    torch.testing.assert_close(baseline.double(), independent_design(x, prelabel), rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_zero_forecast_design_invariant_after_learning_and_frozen_base_parity():
    model = StateConditionedStack(14, 'cuda')
    independent_base = DynamicNIG(14, 'cuda')
    xs, ys = stream(16)
    base_start = len(model.meta.output_names)
    base_stop = base_start + len(model.base.output_names)
    for x, y in zip(xs, ys):
        output = model.update(x, y)
        expected = independent_base.update(x, y)
        torch.testing.assert_close(output[base_start:base_stop], expected, rtol=0, atol=0)
        for actual, reference in zip(model.base.state_tensors(), independent_base.state_tensors(), strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert model.meta.mean.square().sum() > 0
    assert model.context_adam.weights.square().sum() > 0
    before = [value.clone() for value in (model.meta.mean, model.meta.cov)]
    output = model.update(torch.zeros_like(xs[0]), torch.tensor(3., device='cuda'))
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    torch.testing.assert_close(composition_features(xs[0], output[base_start:base_stop]),
                               torch.zeros((3, model.feature_dim), device='cuda'), rtol=0, atol=0)
    for actual, reference in zip((model.meta.mean, model.meta.cov), before):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_captured_matched_context_adam_matches_actual_torch_optimizer():
    model = StateConditionedStack(14, 'cuda')
    bank = model.context_adam
    lrs = stock.Args.adam_lrs
    parameters = [torch.nn.Parameter(torch.zeros(128, device='cuda')) for _ in lrs]
    optimizers = [torch.optim.Adam([parameter], lr=lr, betas=(.9, .999), eps=1e-5,
                                   foreach=False, fused=False)
                  for parameter, lr in zip(parameters, lrs)]
    xs, ys = stream(24)
    x, y = xs[0].clone(), ys[0].clone()
    base_start = len(model.meta.output_names)
    base_stop = base_start + len(model.base.output_names)
    # Independently reconstruct the design from the actual captured pre-label
    # base trajectory: this also detects passing the wrong features or target
    # to ContextAdam inside the composed learner.
    with torch.no_grad(), captured_update(model, x, y) as (graph, output, _):
        for x_row, y_row in zip(xs, ys):
            x.copy_(x_row)
            y.copy_(y_row)
            graph.replay()
            predicted = output.clone()
            prelabel = predicted[base_start:base_stop]
            design = independent_design(x, prelabel)[2].float()
            target = y - prelabel[1]
            for index, (parameter, optimizer) in enumerate(zip(parameters, optimizers)):
                with torch.enable_grad():
                    optimizer.zero_grad(set_to_none=True)
                    prediction = (parameter * design).sum()
                    torch.testing.assert_close(predicted[-len(lrs) + index], prediction + prelabel[1],
                                               rtol=3e-4, atol=2e-6)
                    (.5 * (prediction - target).square()).backward()
                    optimizer.step()
                torch.testing.assert_close(bank.weights[index], parameter, rtol=3e-4, atol=2e-6)
                torch.testing.assert_close(bank.m[index], optimizer.state[parameter]['exp_avg'],
                                           rtol=3e-4, atol=2e-6)
                torch.testing.assert_close(bank.v[index], optimizer.state[parameter]['exp_avg_sq'],
                                           rtol=3e-4, atol=2e-7)
                torch.testing.assert_close(bank.steps.cpu(), optimizer.state[parameter]['step'].cpu(),
                                           rtol=0, atol=0)
