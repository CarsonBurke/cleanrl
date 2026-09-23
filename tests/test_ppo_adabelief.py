"""Run CUDA optimizer regressions through mlq; no training smoke runs."""

import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_normres_indclip_v4_adabelief_50M_v1 import AdaBelief as NormresAdaBelief
from cleanrl.ppo_continuous_action_adabelief_v1 import AdaBelief as BaseAdaBelief
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    return device


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(params=[NormresAdaBelief, BaseAdaBelief], ids=["normres", "base"])
def optimizer_class(request):
    return request.param


def oracle(weight, gradient, moment, variance, step, lr, eps):
    moment = 0.9 * moment + 0.1 * gradient
    variance = 0.999 * variance + 0.001 * (gradient - moment) ** 2 + eps
    weight = weight - lr * (moment / (1 - 0.9 ** step)) / (np.sqrt(variance / (1 - 0.999 ** step)) + eps)
    return weight, moment, variance


def test_compiled_updates_match_paper_with_live_lr_and_no_recompilation(optimizer_class):
    from torch._dynamo.backends.registry import lookup_backend

    compilations = 0
    inductor = lookup_backend("inductor")

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return inductor(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    real_compile = torch.compile
    initial = np.array([[0.2, -0.8, 1.3], [-0.4, 0.7, 0.1]], dtype=np.float32)
    parameter = torch.nn.Parameter(torch.tensor(initial, device="cuda"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=backend, **kwargs))
        optimizer = optimizer_class([parameter], lr=0.0096, eps=1e-8)
    weight = initial.astype(np.float64)
    moment, variance = np.zeros_like(weight), np.zeros_like(weight)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(20):
            # Persistent, reversing, tiny and zero gradients expose m_t vs m_{t-1},
            # bias correction, and the accumulated epsilon floor independently.
            gradient = np.array([[0.3, (-1.0) ** index, 1e-7], [0.0, 0.2 * index, -0.1]], dtype=np.float32)
            lr = 0.0096 * (1 - index / 20) if index != 12 else 0.0
            parameter.grad = torch.tensor(gradient, device="cuda")
            optimizer.set_lr(lr)
            previous = torch.cuda.get_sync_debug_mode()
            try:
                if index:
                    torch.cuda.set_sync_debug_mode("error")
                optimizer.step()
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            weight, moment, variance = oracle(weight, gradient.astype(np.float64), moment, variance, index + 1, np.float32(lr), 1e-8)
            np.testing.assert_allclose(parameter.detach().cpu().numpy(), weight, rtol=3e-5, atol=3e-6)
    assert compilations == 1


def test_missing_gradient_does_not_advance_bias_correction_or_momentum(optimizer_class):
    parameter = torch.nn.Parameter(torch.tensor([0.5], device="cuda"))
    delayed = torch.nn.Parameter(torch.tensor([0.5], device="cuda"))
    optimizer = optimizer_class([parameter, delayed], lr=0.0096, eps=1e-8, compile=False)
    parameter.grad = torch.tensor([0.2], device="cuda")
    optimizer.step()
    first_update = parameter.detach().clone()
    parameter.grad = None
    for _ in range(4):
        optimizer.step()
    torch.testing.assert_close(parameter, first_update, rtol=0, atol=0)
    delayed.grad = torch.tensor([0.2], device="cuda")
    optimizer.step()
    torch.testing.assert_close(delayed, first_update, rtol=0, atol=0)
    # A zero gradient IS an observation: existing momentum still moves the weight.
    parameter.grad = torch.zeros_like(parameter)
    delayed.grad = None
    optimizer.step()
    expected, moment, variance = oracle(np.array([0.5]), np.array([0.2]), np.zeros(1), np.zeros(1), 1, np.float32(0.0096), 1e-8)
    expected, _, _ = oracle(expected, np.zeros(1), moment, variance, 2, np.float32(0.0096), 1e-8)
    np.testing.assert_allclose(parameter.detach().cpu().numpy(), expected, rtol=3e-5, atol=3e-6)
