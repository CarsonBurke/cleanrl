"""Queue these CUDA Adam/prediction-controller checks through mlq."""

import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_predict_check_adam_v1 import PredictCheckAdam, adapt_learning_rate
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_compiled_step_matches_default_adam_and_independent_momentum_prediction():
    from torch._inductor.compile_fx import compile_fx

    compilations = 0

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return compile_fx(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    real_compile = torch.compile
    initial = [np.array([[0.2, -0.8], [1.3, 0.0]], dtype=np.float32), np.array([0.3], dtype=np.float32)]
    params = [torch.nn.Parameter(torch.tensor(value, device="cuda")) for value in initial]
    references = [torch.nn.Parameter(parameter.detach().clone()) for parameter in params]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=backend, **kwargs))
        optimizer = PredictCheckAdam(params)
    reference = torch.optim.Adam(references, lr=3e-4, eps=1e-5, fused=True)
    means = [np.zeros_like(value, dtype=np.float64) for value in initial]
    seconds = [value.copy() for value in means]
    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(24):
            lr = float(np.float32(3e-4 * (1 - index / 24))) if index != 12 else 0.0
            optimizer.param_groups[0]["lr"].fill_(lr)
            reference.param_groups[0]["lr"] = lr
            gradients = [
                np.array([[0.3, (-1.0) ** index], [1e-7, 0.0]], dtype=np.float32),
                np.array([4.0 if index > 10 else 0.01], dtype=np.float32),
            ]
            prediction = 0.0
            for slot, (parameter, ref, gradient) in enumerate(zip(params, references, gradients)):
                parameter.grad = torch.tensor(gradient, device="cuda")
                ref.grad = parameter.grad.clone()
                means[slot] = 0.9 * means[slot] + 0.1 * gradient.astype(np.float64)
                seconds[slot] = 0.999 * seconds[slot] + 0.001 * gradient.astype(np.float64) ** 2
                corrected = means[slot] / (1 - 0.9 ** (index + 1))
                direction = corrected / (np.sqrt(seconds[slot] / (1 - 0.999 ** (index + 1))) + 1e-5)
                prediction += lr * np.sum(corrected * direction)
            previous = torch.cuda.get_sync_debug_mode()
            try:
                if index:
                    torch.cuda.set_sync_debug_mode("error")
                optimizer.step()
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            reference.step()
            for parameter, ref, gradient in zip(params, references, gradients):
                torch.testing.assert_close(parameter, ref, rtol=2e-5, atol=2e-6)
                assert parameter.grad is not None
                np.testing.assert_array_equal(parameter.grad.cpu().numpy(), gradient)
            np.testing.assert_allclose(optimizer.predicted_decrease.cpu().numpy(), prediction, rtol=3e-5, atol=1e-10)
    assert compilations == 1


def test_controller_thresholds_ema_and_undefined_prediction():
    scalar = lambda value: torch.tensor(value, device="cuda", dtype=torch.float64)
    lr, total, weight = scalar(0.01), scalar(0), scalar(0)
    # Exact threshold equality holds the LR; above/below thresholds change it.
    expected_lr = 0.01
    for ratio, factor in ((0.75, 1.0), (0.25, 1.0), (0.9, 1.05), (0.1, 0.8), (-0.5, 0.8)):
        used = lr.clone()
        with torch.no_grad():
            stats = adapt_learning_rate(lr, total, weight, scalar(2 * ratio), scalar(0), scalar(2), used, 0.0)
        expected_lr *= factor
        assert lr.item() == pytest.approx(expected_lr)
        assert stats[3].item() == pytest.approx(ratio)
    # With EMA history, one bad observation need not immediately shrink the LR.
    lr, total, weight = scalar(0.01), scalar(0), scalar(0)
    with torch.no_grad():
        adapt_learning_rate(lr, total, weight, scalar(1), scalar(0), scalar(1), lr.clone(), 0.9)
        stats = adapt_learning_rate(lr, total, weight, scalar(0), scalar(0), scalar(1), lr.clone(), 0.9)
    assert stats[3].item() == pytest.approx(0.9 / 1.9)
    assert lr.item() == pytest.approx(0.0105)
    before = [tensor.clone() for tensor in (lr, total, weight)]
    with torch.no_grad():
        stats = adapt_learning_rate(lr, total, weight, scalar(1), scalar(1), scalar(0), lr.clone(), 0.9)
    for actual, expected in zip((lr, total, weight), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats[7].item() == 0


def test_real_quadratic_check_adapts_lr_without_recompilation_or_cuda_sync():
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = PredictCheckAdam([parameter], lr=0.4, betas=(0.0, 0.0), trust_beta=0.5)
    expected_sum, expected_weight = 0.0, 0.0
    growing = shrinking = False
    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(18):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * parameter.square().sum()
            before = loss.detach().clone()
            loss.backward()
            used_lr = optimizer.param_groups[0]["lr"].item()
            optimizer.step()
            after = 0.5 * parameter.detach().square().sum()
            previous = torch.cuda.get_sync_debug_mode()
            try:
                if index:
                    torch.cuda.set_sync_debug_mode("error")
                stats = optimizer.observe_loss(before, after)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            prediction = optimizer.predicted_decrease.item()
            ratio = (before.item() - after.item()) / prediction
            expected_sum = 0.5 * expected_sum + 0.5 * ratio
            expected_weight = 0.5 * expected_weight + 0.5
            rho = expected_sum / expected_weight
            factor = 1.05 if rho > 0.75 else 0.8 if rho < 0.25 else 1.0
            growing |= factor > 1
            shrinking |= factor < 1
            assert stats[3].item() == pytest.approx(rho, rel=2e-5, abs=2e-5)
            assert optimizer.param_groups[0]["lr"].item() == pytest.approx(used_lr * factor, rel=2e-6)
    assert growing and shrinking


def test_missing_gradients_do_not_reuse_previous_prediction_or_advance_adam_history():
    parameter = torch.nn.Parameter(torch.tensor([0.5], device="cuda"))
    delayed = torch.nn.Parameter(parameter.detach().clone())
    optimizer = PredictCheckAdam([parameter, delayed], compile=False)
    parameter.grad = torch.tensor([0.2], device="cuda")
    optimizer.step()
    expected = parameter.detach().clone()
    parameter.grad = None
    for _ in range(3):
        optimizer.step()
        assert optimizer.predicted_decrease.item() == 0
        optimizer.observe_loss(torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda"))
    delayed.grad = torch.tensor([0.2], device="cuda")
    optimizer.step()
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(delayed, expected, rtol=0, atol=0)
