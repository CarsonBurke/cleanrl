"""Queue these CUDA optimizer regressions through mlq, not direct pytest.

The numerical oracle uses independent NumPy float64 arithmetic on the host;
all optimizer execution, including Inductor coverage, uses CUDA matrices.
"""

import copy
import io

import numpy as np
import pytest
import torch

from cleanrl.shared.adamh import AdamH
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    return device


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


INITIAL = np.array([[0.2, -0.8, 1.3], [-0.4, 0.7, 0.1]], dtype=np.float64)


def parameter(value=INITIAL):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device="cuda"))


def gradient(index):
    return np.sin(np.arange(6, dtype=np.float64).reshape(2, 3) + 0.7 * index) + 0.13 * index


def oracle(weight, grad, moment, variance, step, radius, lr, betas=(0.9, 0.999), eps=1e-5):
    """Literal paper equation, not shared torch optimizer helper code."""
    if grad is None:
        return weight.copy(), moment.copy(), variance.copy(), step
    beta1, beta2 = betas
    step += 1
    moment = beta1 * moment + (1 - beta1) * grad
    variance = beta2 * variance + (1 - beta2) * grad ** 2
    direction = (moment / (1 - beta1 ** step)) / (np.sqrt(variance / (1 - beta2 ** step)) + eps)
    length = np.sqrt(np.sum(direction ** 2))
    if length == 0 or radius == 0 or lr == 0:
        return weight.copy(), moment, variance, step
    trial = weight - lr * radius * direction / length
    trial_length = np.sqrt(np.sum(trial ** 2))
    if trial_length == 0:
        return weight.copy(), moment, variance, step
    return radius * trial / trial_length, moment, variance, step


def assert_array(actual, expected, tolerance=2e-11):
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=tolerance, atol=tolerance)


def test_multistep_matches_independent_bias_corrected_adam_hyperball_oracle():
    p = parameter()
    opt = AdamH([p], compile=False)
    weight = INITIAL.copy()
    moment = np.zeros_like(weight)
    variance = np.zeros_like(weight)
    radius, step = np.linalg.norm(weight), 0
    for index, lr in enumerate((0.018, 0.031, 0.009, 0.0, 0.07, 0.005)):
        grad = gradient(index)
        p.grad = torch.tensor(grad, device="cuda")
        opt.set_lr(lr)
        opt.step()
        weight, moment, variance, step = oracle(weight, grad, moment, variance, step, radius, lr)
        assert_array(p, weight)
        assert_array(torch.linalg.vector_norm(p), radius)


def test_radius_is_captured_at_construction_not_first_step_or_current_weight():
    p = parameter()
    opt = AdamH([p], lr=0.13, compile=False)
    radius = np.linalg.norm(INITIAL)
    moment, variance, step = np.zeros_like(INITIAL), np.zeros_like(INITIAL), 0
    # Exaggerate roundoff/external perturbation so recapturing the current
    # radius is a deterministic bug, rather than relying on accumulated drift.
    for index, scale in enumerate((1.25, 0.7, 1.1)):
        with torch.no_grad():
            p.mul_(scale)
        before = p.detach().cpu().numpy().copy()
        grad = gradient(index)
        p.grad = torch.tensor(grad, device="cuda")
        expected, moment, variance, step = oracle(before, grad, moment, variance, step, radius, 0.13)
        opt.step()
        assert_array(p, expected)
        assert_array(torch.linalg.vector_norm(p), radius)


def test_whole_matrix_update_is_scale_and_transpose_equivariant():
    scale, eps = 7.0, 1e-4
    base, scaled, transposed = parameter(), parameter(INITIAL * scale), parameter(INITIAL.T.copy())
    opts = [AdamH([base], lr=0.13, eps=eps, compile=False),
            AdamH([scaled], lr=0.13, eps=eps / scale, compile=False),
            AdamH([transposed], lr=0.13, eps=eps, compile=False)]
    for index in range(5):
        grad = gradient(index)
        # A scale-invariant loss scales gradients inversely. Epsilon must
        # scale too for exact Adam equivariance; fixed epsilon breaks it.
        for p, opt, values in zip((base, scaled, transposed), opts, (grad, grad / scale, grad.T.copy())):
            p.grad = torch.tensor(values, device="cuda")
            opt.step()
        torch.testing.assert_close(scaled / scale, base, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(transposed.T, base, rtol=2e-11, atol=2e-11)


def test_zero_gradient_advances_time_but_none_gradient_does_not():
    p, never_used = parameter(), parameter(INITIAL + 0.4)
    opt = AdamH([p, never_used], lr=0.2, compile=False)
    before_unused = never_used.detach().clone()
    initial_unused_state = copy.deepcopy(opt.state[never_used])
    weight, moment, variance = INITIAL.copy(), np.zeros_like(INITIAL), np.zeros_like(INITIAL)
    radius, step = np.linalg.norm(INITIAL), 0
    for grad in (np.zeros_like(INITIAL), None, gradient(1), None, np.zeros_like(INITIAL), gradient(3)):
        before = p.detach().clone()
        state_before = copy.deepcopy(opt.state[p])
        p.grad = None if grad is None else torch.tensor(grad, device="cuda")
        opt.step()
        weight, moment, variance, step = oracle(weight, grad, moment, variance, step, radius, 0.2)
        assert_array(p, weight)
        if grad is None:
            torch.testing.assert_close(p, before, rtol=0, atol=0)
            for key, value in state_before.items():
                torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0)
        torch.testing.assert_close(never_used, before_unused, rtol=0, atol=0)
        assert opt.state[never_used].keys() == initial_unused_state.keys()
        for key, value in initial_unused_state.items():
            torch.testing.assert_close(opt.state[never_used][key], value, rtol=0, atol=0)


def test_zero_update_is_exact_noop_even_if_matrix_was_perturbed():
    p = parameter()
    opt = AdamH([p], compile=False)
    with torch.no_grad():
        p.mul_(1.1)
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    # The next update must nevertheless use t=2 bias correction.
    p.grad = torch.tensor(gradient(1), device="cuda")
    expected, _, _, _ = oracle(before.cpu().numpy(), gradient(1), np.zeros_like(INITIAL),
                               np.zeros_like(INITIAL), 1, np.linalg.norm(INITIAL), 0.018)
    opt.step()
    assert_array(p, expected)


def test_zero_radius_and_undefined_trial_retain_the_previous_point():
    origin, p = parameter(np.zeros((1, 1))), parameter(np.array([[2.0]]))
    opt = AdamH([origin, p], lr=1.0, betas=(0.0, 0.0), compile=False)
    origin.grad, p.grad = torch.ones_like(origin), torch.ones_like(p)
    opt.step()  # trial for p is exactly 2 - 1 * 2 * 1 = 0.
    assert_array(origin, np.zeros((1, 1)), tolerance=0)
    assert_array(p, np.array([[2.0]]), tolerance=0)
    opt.set_lr(3.0)
    opt.step()
    assert_array(origin, np.zeros((1, 1)), tolerance=0)
    assert_array(p, np.array([[-2.0]]), tolerance=0)


def test_checkpoint_resume_restores_fixed_radius_moments_time_and_group_options():
    p, delayed = parameter(), parameter(INITIAL + 0.3)
    opt = AdamH([p, delayed], lr=0.14, betas=(0.7, 0.95), eps=0.003, compile=False)
    for index in range(3):
        p.grad = torch.tensor(gradient(index), device="cuda")
        opt.step()
    opt.set_lr(0.037)
    saved = io.BytesIO()
    torch.save(opt.state_dict(), saved)
    saved.seek(0)
    # Constructor sees a deliberately different radius; loading must restore
    # the saved initial radius, not compute one from the newly supplied model.
    resumed, resumed_delayed = parameter(INITIAL * 2), parameter(INITIAL * 3)
    restored = AdamH([resumed, resumed_delayed], lr=0.8, betas=(0.1, 0.2), eps=0.5, compile=False)
    restored.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True))
    with torch.no_grad():
        resumed.copy_(p)
        resumed_delayed.copy_(delayed)
    for index in range(3, 7):
        for a, b, offset in ((p, resumed, 0), (delayed, resumed_delayed, 2)):
            a.grad = torch.tensor(gradient(index + offset), device="cuda")
            b.grad = a.grad.clone()
        opt.step()
        restored.step()
        torch.testing.assert_close(resumed, p, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(resumed_delayed, delayed, rtol=2e-11, atol=2e-11)


def test_compiled_lr_changes_match_oracle_without_recompilation():
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("inductor")
    real_compile = torch.compile
    # Count actual Inductor compilations without replacing the tensor update
    # with a fake backend. error_on_recompile also rejects LR/step guards.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=counter, **kwargs))
        p = parameter()
        opt = AdamH([p])
    weight, moment, variance = INITIAL.copy(), np.zeros_like(INITIAL), np.zeros_like(INITIAL)
    radius, step = np.linalg.norm(INITIAL), 0
    with torch._dynamo.config.patch(error_on_recompile=True):
        for index, lr in enumerate((0.018, 0.04, 0.005, 0.0, 0.027)):
            grad = gradient(index)
            p.grad = torch.tensor(grad, device="cuda")
            opt.set_lr(lr)
            # Any accidental .item()/tensor truth test in the optimizer must
            # fail; assertions/CPU oracle copies happen outside this scope.
            if index == 0:
                opt.step()  # Inductor's initial compilation may synchronize.
            else:
                with _no_cuda_sync():
                    opt.step()
            weight, moment, variance, step = oracle(weight, grad, moment, variance, step, radius, lr)
            assert_array(p, weight)
    assert counter.frame_count == 1


class _no_cuda_sync:
    def __enter__(self):
        self.previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exc_type, exc, traceback):
        torch.cuda.set_sync_debug_mode(self.previous)


def test_rejects_nonmatrix_parameters_and_sparse_gradients():
    for values in (torch.ones(3, device="cuda"), torch.ones(2, 3, 4, device="cuda"),
                   torch.empty(0, 3, device="cuda")):
        with pytest.raises(ValueError, match="matrices"):
            AdamH([torch.nn.Parameter(values)], compile=False)
    for dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(ValueError, match="float32 or float64"):
            AdamH([torch.nn.Parameter(torch.ones(2, 3, device="cuda", dtype=dtype))], compile=False)
    p = parameter()
    opt = AdamH([p], compile=False)
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [1]], device="cuda"),
                                   torch.tensor([1.0], device="cuda", dtype=torch.float64), p.shape)
    before = p.detach().clone()
    with pytest.raises(RuntimeError, match="sparse gradients"):
        opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)
