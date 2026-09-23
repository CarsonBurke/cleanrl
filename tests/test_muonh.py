"""Queue these CUDA optimizer regressions through mlq, not direct pytest.

The numerical oracle uses independent NumPy float64 arithmetic on the host,
including its own Newton-Schulz polynomial; all optimizer execution,
including Inductor coverage, uses CUDA matrices.
"""

import copy
import io

import numpy as np
import pytest
import torch

from cleanrl.shared.muonh import MuonH
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    return device


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


INITIAL = np.array([[0.2, -0.8, 1.3], [-0.4, 0.7, 0.1]], dtype=np.float64)


def parameter(value=INITIAL, dtype=torch.float64):
    return torch.nn.Parameter(torch.tensor(value, dtype=dtype, device="cuda"))


def gradient(index):
    return np.sin(np.arange(6, dtype=np.float64).reshape(2, 3) + 0.7 * index) + 0.13 * index


def newton_schulz(matrix, iterations=5):
    """Independent host copy of the quintic polar iteration, gain sqrt(O * I)."""
    rows, columns = matrix.shape
    transposed = rows > columns
    direction = matrix.T if transposed else matrix
    direction = direction / (np.sqrt(np.sum(direction ** 2)) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(iterations):
        gram = direction @ direction.T
        direction = a * direction + (b * gram + c * (gram @ gram)) @ direction
    if transposed:
        direction = direction.T
    return direction * (np.sqrt(rows * columns) / (np.sqrt(np.sum(direction ** 2)) + 1e-7))


def oracle(weight, grad, moment, radius, lr, momentum=0.9, ns_steps=5):
    """Literal paper equations (11)-(13) plus Algorithm 1, no torch helpers."""
    if grad is None:
        return weight.copy(), moment.copy()
    moment = momentum * moment + (1 - momentum) * grad
    direction = newton_schulz(moment, ns_steps)
    length = np.sqrt(np.sum(direction ** 2))
    if length == 0 or radius == 0 or lr == 0:
        return weight.copy(), moment
    trial = weight - lr * radius * direction / length
    trial_length = np.sqrt(np.sum(trial ** 2))
    if trial_length == 0:
        return weight.copy(), moment
    return radius * trial / trial_length, moment


def assert_array(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=tolerance, atol=tolerance)


def test_multistep_matches_independent_newton_schulz_hyperball_oracle():
    p = parameter()
    opt = MuonH([p], compile=False)
    weight, moment = INITIAL.copy(), np.zeros_like(INITIAL)
    radius = np.linalg.norm(INITIAL)
    for index, lr in enumerate((0.018, 0.031, 0.009, 0.0, 0.07, 0.005)):
        grad = gradient(index)
        p.grad = torch.tensor(grad, device="cuda")
        opt.set_lr(lr)
        opt.step()
        weight, moment = oracle(weight, grad, moment, radius, lr)
        assert_array(p, weight)
        assert_array(opt.state[p]["momentum_buffer"], moment)
        assert_array(torch.linalg.vector_norm(p), radius)


def test_radius_is_captured_at_construction_not_first_step_or_current_weight():
    p = parameter()
    opt = MuonH([p], lr=0.13, compile=False)
    radius, moment = np.linalg.norm(INITIAL), np.zeros_like(INITIAL)
    # Exaggerate roundoff/external perturbation so recapturing the current
    # radius is a deterministic bug, rather than relying on accumulated drift.
    for index, scale in enumerate((1.25, 0.7, 1.1)):
        with torch.no_grad():
            p.mul_(scale)
        before = p.detach().cpu().numpy().copy()
        grad = gradient(index)
        p.grad = torch.tensor(grad, device="cuda")
        expected, moment = oracle(before, grad, moment, radius, 0.13)
        opt.step()
        assert_array(p, expected)
        assert_array(torch.linalg.vector_norm(p), radius)


def test_float32_trunk_shapes_hold_their_radius_over_many_steps():
    generator = torch.Generator(device="cuda").manual_seed(7)
    matrices = [parameter(dtype=torch.float32, value=np.zeros(shape)) for shape in ((43, 64), (64, 43))]
    with torch.no_grad():
        for p in matrices:
            p.normal_(std=64 ** -0.5, generator=generator)
    opt = MuonH(matrices, compile=False)
    radii = [opt.state[p]["radius"].clone() for p in matrices]
    worst = 0.0
    for step in range(200):
        opt.set_lr(0.018 * (1.0 - step / 200))
        for p in matrices:
            p.grad = torch.randn(p.shape, device="cuda", generator=generator)
        opt.step()
        for p, radius in zip(matrices, radii):
            error = ((torch.linalg.vector_norm(p) - radius).abs() / radius).item()
            worst = max(worst, error)
    print(f"max float32 radius relative error over 200 steps: {worst:.3e}")
    assert worst <= 1e-6


def test_update_is_invariant_to_a_positive_gradient_rescale():
    scale = 37.0
    base, scaled = parameter(), parameter()
    opts = [MuonH([base], lr=0.11, compile=False), MuonH([scaled], lr=0.11, compile=False)]
    worst = 0.0
    for index in range(6):
        # msign(c M) = msign(M) and Normalize kills every surviving constant,
        # so a constant gradient rescale must not change the trajectory. The
        # residual is polar_direction's +1e-7 Frobenius pre-normalization guard.
        for p, opt, values in zip((base, scaled), opts, (gradient(index), gradient(index) * scale)):
            p.grad = torch.tensor(values, device="cuda")
            opt.step()
        worst = max(worst, (scaled - base).abs().max().item() / base.abs().max().item())
    print(f"max relative trajectory deviation under a {scale}x gradient rescale: {worst:.3e}")
    assert worst <= 1e-6


def test_zero_gradient_decays_momentum_but_a_missing_gradient_is_a_noop():
    p, never_used = parameter(), parameter(INITIAL + 0.4)
    opt = MuonH([p, never_used], lr=0.2, compile=False)
    before_unused = never_used.detach().clone()
    initial_unused_state = copy.deepcopy(opt.state[never_used])
    weight, moment = INITIAL.copy(), np.zeros_like(INITIAL)
    radius = np.linalg.norm(INITIAL)
    for grad in (np.zeros_like(INITIAL), None, gradient(1), None, np.zeros_like(INITIAL), gradient(3)):
        before = p.detach().clone()
        state_before = copy.deepcopy(opt.state[p])
        p.grad = None if grad is None else torch.tensor(grad, device="cuda")
        opt.step()
        weight, moment = oracle(weight, grad, moment, radius, 0.2)
        assert_array(p, weight)
        assert_array(torch.linalg.vector_norm(p), radius)
        if grad is None:
            torch.testing.assert_close(p, before, rtol=0, atol=0)
            for key, value in state_before.items():
                torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0)
        torch.testing.assert_close(never_used, before_unused, rtol=0, atol=0)
        assert opt.state[never_used].keys() == initial_unused_state.keys()
        for key, value in initial_unused_state.items():
            torch.testing.assert_close(opt.state[never_used][key], value, rtol=0, atol=0)


def test_zero_momentum_matrix_and_zero_lr_are_exact_noops():
    p = parameter()
    opt = MuonH([p], compile=False)
    with torch.no_grad():
        p.mul_(1.1)
    before = p.detach().clone()
    # First step with an exactly zero gradient: momentum stays zero, so the
    # Newton-Schulz direction is zero and there is no direction to take.
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    torch.testing.assert_close(opt.state[p]["momentum_buffer"], torch.zeros_like(p), rtol=0, atol=0)
    opt.set_lr(0.0)
    p.grad = torch.tensor(gradient(2), device="cuda")
    opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    # A zero LR still advanced the momentum EMA; the next real step uses it.
    expected, _ = oracle(before.cpu().numpy(), gradient(3), 0.1 * gradient(2), np.linalg.norm(INITIAL), 0.018)
    opt.set_lr(0.018)
    p.grad = torch.tensor(gradient(3), device="cuda")
    opt.step()
    assert_array(p, expected)


def test_zero_radius_and_undefined_trial_retain_the_previous_point():
    origin, p = parameter(np.zeros((1, 1))), parameter(np.array([[2.0]]))
    opt = MuonH([origin, p], lr=1.0, momentum=0.0, compile=False)
    origin.grad, p.grad = torch.ones_like(origin), torch.ones_like(p)
    opt.step()  # trial for p is exactly 2 - 1 * 2 * 1 = 0.
    assert_array(origin, np.zeros((1, 1)), tolerance=0)
    assert_array(p, np.array([[2.0]]), tolerance=0)
    opt.set_lr(3.0)
    opt.step()
    assert_array(origin, np.zeros((1, 1)), tolerance=0)
    assert_array(p, np.array([[-2.0]]), tolerance=0)


def test_checkpoint_resume_restores_fixed_radius_momentum_and_group_options():
    p, delayed = parameter(), parameter(INITIAL + 0.3)
    opt = MuonH([p, delayed], lr=0.14, momentum=0.7, ns_steps=3, compile=False)
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
    restored = MuonH([resumed, resumed_delayed], lr=0.8, momentum=0.1, ns_steps=5, compile=False)
    restored.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True))
    assert restored.param_groups[0]["ns_steps"] == 3
    assert restored.param_groups[0]["lr"].device == resumed.device
    with torch.no_grad():
        resumed.copy_(p)
        resumed_delayed.copy_(delayed)
    for index in range(3, 7):
        for a, b, offset in ((p, resumed, 0), (delayed, resumed_delayed, 2)):
            a.grad = torch.tensor(gradient(index + offset), device="cuda")
            b.grad = a.grad.clone()
        opt.step()
        restored.step()
        torch.testing.assert_close(resumed, p, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(resumed_delayed, delayed, rtol=1e-10, atol=1e-10)


def test_compiled_lr_changes_match_oracle_without_recompilation():
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("inductor")
    real_compile = torch.compile
    # Count actual Inductor compilations without replacing the tensor update
    # with a fake backend. error_on_recompile also rejects LR guards.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=counter, **kwargs))
        p = parameter()
        opt = MuonH([p])
    weight, moment = INITIAL.copy(), np.zeros_like(INITIAL)
    radius = np.linalg.norm(INITIAL)
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
            weight, moment = oracle(weight, grad, moment, radius, lr)
            assert_array(p, weight)
    assert counter.frame_count == 1


class _no_cuda_sync:
    def __enter__(self):
        self.previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exc_type, exc, traceback):
        torch.cuda.set_sync_debug_mode(self.previous)


def test_rejects_nonmatrix_parameters_hyperparameters_and_sparse_gradients():
    for values in (torch.ones(3, device="cuda"), torch.ones(2, 3, 4, device="cuda"),
                   torch.empty(0, 3, device="cuda")):
        with pytest.raises(ValueError, match="matrices"):
            MuonH([torch.nn.Parameter(values)], compile=False)
    for dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(ValueError, match="float32 or float64"):
            MuonH([torch.nn.Parameter(torch.ones(2, 3, device="cuda", dtype=dtype))], compile=False)
    with pytest.raises(ValueError, match="momentum"):
        MuonH([parameter()], momentum=1.0, compile=False)
    with pytest.raises(ValueError, match="Newton-Schulz"):
        MuonH([parameter()], ns_steps=0, compile=False)
    p = parameter()
    opt = MuonH([p], compile=False)
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [1]], device="cuda"),
                                     torch.tensor([1.0], device="cuda", dtype=torch.float64), p.shape)
    before = p.detach().clone()
    with pytest.raises(RuntimeError, match="sparse gradients"):
        opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)
