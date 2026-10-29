"""CUDA optimizer contracts. Submit these checks through mlq."""
import copy
import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_anvil2_v4 import (
    ANVIL_MAPS, MatrixOptimizer, matrix_update, spectral_cascade,
)


@pytest.fixture(scope='session')
def numerical_evidence():
    measurements = []
    yield measurements
    path = Path(__file__).resolve().parents[1] / 'benchmarks/sepclip_anvil2_v4/contracts.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = Path(__file__).resolve().parents[2] / 'modded-nanogpt'
    revision = (subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
                if reference.exists() else None)
    path.write_text(json.dumps({'measurements': measurements,
                               'reference_revision': revision,
                               'torch_version': torch.__version__,
                               'matmul_precision': torch.get_float32_matmul_precision(),
                               'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                               'note': 'Measurements are evidence, not a substitute for the mlq test outcome.'}, indent=2) + '\n')


@pytest.mark.parametrize('shape', [(43, 17), (43, 64), (64, 43)])
@pytest.mark.parametrize('condition_number', [10., 1e4])
def test_cascade_matches_independent_scalar_spectral_oracle(shape, condition_number, numerical_evidence):
    torch.manual_seed(1)
    rank = min(shape)
    u, _ = torch.linalg.qr(torch.randn(shape[0], rank, device='cuda', dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(shape[1], rank, device='cuda', dtype=torch.float64))
    s = torch.logspace(-math.log10(condition_number), 0, rank, device='cuda', dtype=torch.float64)
    g = ((u * s) @ v.T).float()
    spectrum = s / (s.norm() * 1.05 + 1e-6)
    for a, b, c in ANVIL_MAPS:
        spectrum = a * spectrum + b * spectrum**3 + c * spectrum**5
    expected = ((u * spectrum) @ v.T).float()
    for precision in ('fp32', 'bf16'):
        actual = spectral_cascade(g, 'anvil2', precision)
        relative_error = (actual - expected).norm() / expected.norm()
        numerical_evidence.append(dict(contract='scalar_spectral_oracle', shape=shape,
                                       condition_number=condition_number,
                                       precision=precision, relative_error=float(relative_error),
                                       tolerance=.001 if precision == 'fp32' else None))
        if precision == 'fp32':
            assert relative_error < .001, relative_error
        else:
            # Compare FP64 dot products rounded at source BF16 boundaries.
            # Ill-conditioned cascades amplify accumulation-order differences:
            # record that sensitivity, and require SAME-input upstream parity
            # separately. Enforce close agreement here on well-conditioned data.
            x = g.bfloat16()
            if shape[0] > shape[1]:
                x = x.T
            gram = (x.double() @ x.double().T).bfloat16()
            denominator = gram.diagonal().float().sum().sqrt() * 1.05 + 1e-6
            x = (x.float() / denominator).bfloat16()
            gram = (gram.float() / denominator.square()).bfloat16()
            for index, (a, b, c) in enumerate(ANVIL_MAPS):
                if index:
                    gram = (x.double() @ x.double().T).bfloat16()
                polynomial = (b * gram.double() + c * (gram.double() @ gram.double())).bfloat16()
                x = (a * x.double() + polynomial.double() @ x.double()).bfloat16()
            rounded_expected = (x.T if shape[0] > shape[1] else x).float()
            boundary_error = (actual - rounded_expected).norm() / rounded_expected.norm()
            numerical_evidence.append(dict(contract='independent_bf16_boundaries', shape=shape,
                                           condition_number=condition_number,
                                           relative_error=float(boundary_error),
                                           tolerance=.03 if condition_number == 10. else None))
            if condition_number == 10.:
                assert boundary_error < .03, boundary_error
        zero = spectral_cascade(torch.zeros_like(g), 'anvil2', precision)
        assert torch.count_nonzero(zero) == 0


def test_reference_rail_schedule_and_phase_scaling():
    for total in (1194, 4880):
        assert MatrixOptimizer.rail_schedule(0, total) == (.85, .85, 1.)
        before = MatrixOptimizer.rail_schedule(513 * total / 1194, total)
        after = MatrixOptimizer.rail_schedule(514 * total / 1194, total)
        assert before == (.93, .93, 1.)
        assert after == (.93, .85, .4385)
        assert MatrixOptimizer.rail_schedule(total, total) == pytest.approx((.85, .85, .4385))


@pytest.mark.parametrize('method', ['normuon', 'anvil2'])
def test_matrix_update_preserves_its_fp32_gradient_input(method):
    torch.manual_seed(1)
    p = torch.randn(1, 43, 64, device='cuda')
    gradient = torch.randn_like(p)
    saved = gradient.clone()
    velocity = torch.zeros(2, *p.shape, device='cuda')
    second = torch.zeros(1, 1, 64, device='cuda')
    scalars = torch.tensor([.005, .93, .85, .4385], device='cuda')
    matrix_update(p, velocity, second, gradient, scalars, .9, 1., method, 'bf16', .95)
    torch.testing.assert_close(gradient, saved, atol=0, rtol=0)


@pytest.mark.parametrize('method', ['normuon', 'anvil2'])
def test_compiled_checkpoint_continuation_and_gradient_ownership(method):
    torch.manual_seed(1)
    weights = [torch.nn.Parameter(torch.randn(*shape, device='cuda'))
               for shape in [(43, 17), (43, 17), (43, 64), (64, 43)]]
    opt = MatrixOptimizer(weights, method=method, total_steps=8)
    for _ in range(4):
        for p in weights:
            p.grad = torch.randn_like(p)
        opt.step()
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    other = MatrixOptimizer(clones, method=method, total_steps=8)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for lr in (.0025, .001):
        opt.set_lr(lr)
        other.set_lr(lr)
        saved = []
        for p, clone in zip(weights, clones):
            grad = torch.randn_like(p)
            saved.append(grad.clone())
            p.grad, clone.grad = grad.clone(), grad.clone()
        opt.step()
        other.step()
        for p, clone, grad in zip(weights, clones, saved):
            torch.testing.assert_close(p, clone, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
        for old, new in zip(opt.stacks, other.stacks):
            assert new[2] is other.state[new[1][0]]
            for key in ('velocity', 'second'):
                torch.testing.assert_close(old[2][key], new[2][key], atol=0, rtol=0)


def test_normuon_v4_preserves_v3_updates():
    from cleanrl.ppo_continuous_action_sepclip_longaxis_normuon_v3 import PolarExpressNorMuon
    torch.manual_seed(1)
    weights = [torch.nn.Parameter(torch.randn(*shape, device='cuda'))
               for shape in [(43, 17), (43, 64), (64, 43)]]
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    old = PolarExpressNorMuon(weights, compile=False)
    new = MatrixOptimizer(clones, method='normuon', beta2=.95, compile=False)
    for _ in range(5):
        for p, q in zip(weights, clones):
            grad = torch.randn_like(p)
            p.grad, q.grad = grad.clone(), grad.clone()
        old.step()
        new.step()
        for p, q in zip(weights, clones):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('shape', [(2, 43, 17), (2, 43, 64), (2, 64, 43)])
@pytest.mark.parametrize('input_kind', ['gaussian', 'ill_conditioned'])
def test_direct_upstream_bf16_cascade_parity(shape, input_kind, numerical_evidence, monkeypatch):
    import sys
    from pathlib import Path
    reference = Path(__file__).resolve().parents[2] / 'modded-nanogpt'
    if not reference.exists():
        pytest.skip('sibling modded-nanogpt reference unavailable')
    sys.path.insert(0, str(reference))
    from track_1_short.optim import anvil
    launch_adaptation = None
    if shape[-2] < shape[-1]:
        # Upstream's H100 XXT launch needs 147456 shared bytes; this GPU has
        # 101376. Use the unchanged upstream kernel with smaller launch tiles.
        import triton
        from track_1_short.perf.kernels.polar_express import XXT_kernel

        def small_xxt(a, out):
            rows, columns = a.shape[-2:]
            grid = (a.size(0) * triton.cdiv(rows, 32) ** 2,)
            XXT_kernel[grid](
                A_ptr=a, C_ptr=out, M=rows, K=columns,
                a_stride_b=a.stride(0), a_stride_r=a.stride(-2), a_stride_c=a.stride(-1),
                c_stride_b=out.stride(0), c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
                BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8, LOWER_UPPER=1, num_stages=2, num_warps=4)
            return out

        monkeypatch.setattr(anvil, 'XXT', small_xxt)
        launch_adaptation = dict(original_tiles=[64, 128, 128], original_stages=4, original_warps=8,
                                 adapted_tiles=[32, 32, 32], adapted_stages=2, adapted_warps=4,
                                 original_shared_bytes=147456, gpu_limit_shared_bytes=101376,
                                 kernel_body='Unchanged upstream XXT_kernel')
    torch.manual_seed(1)
    if input_kind == 'gaussian':
        grad = torch.randn(*shape, device='cuda').bfloat16()
    else:
        rank = min(shape[-2:])
        u, _ = torch.linalg.qr(torch.randn(shape[1], rank, device='cuda', dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(shape[2], rank, device='cuda', dtype=torch.float64))
        singular_values = torch.logspace(-4, 0, rank, device='cuda', dtype=torch.float64)
        # Identical FP32 input to the scalar-oracle case, repeated in the bank.
        grad = ((u * singular_values) @ v.T).float().bfloat16()[None].repeat(shape[0], 1, 1)
    zero = torch.tensor(0., device='cuda')
    one = torch.tensor(1., device='cuda')
    velocity = torch.zeros(2, *shape, device='cuda')
    expected = anvil.anvil_cascade(grad, velocity, zero, False, zero, one).float()
    actual = spectral_cascade(grad.float(), 'anvil2', 'bf16')
    error = (actual - expected).norm() / expected.norm()
    numerical_evidence.append(dict(contract='direct_upstream_cascade', shape=shape,
                                   input_kind=input_kind,
                                   precision='bf16', relative_error=float(error), tolerance=.03,
                                   reference_launch_adaptation=launch_adaptation))
    print(f'upstream cascade {shape}: relative error={float(error):.6g}')
    assert error < .03, error


def test_twin_rail_and_equalizer_against_independent_arithmetic():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(43, 64, device='cuda'))
    opt = MatrixOptimizer([p], compile=False, total_steps=4)
    velocity = torch.zeros(2, 1, 43, 64, device='cuda')
    second = torch.zeros(1, 1, 64, device='cuda')
    for step in range(4):
        g = torch.randn_like(p)
        p.grad = g.clone()
        before = p.detach().clone()
        lookahead, fast_beta, weight = opt.rail_schedule(step, 4)
        # Independent FP32 EMA; near bf16 cast boundaries only the state and
        # resulting equalizer are compared to production's actual cascade.
        velocity[0] = fast_beta * velocity[0] + (1 - fast_beta) * g
        velocity[1] = .98 * velocity[1] + .02 * g
        opt.step()
        state = opt.state[p]
        torch.testing.assert_close(state['velocity'], velocity, rtol=2e-6, atol=5e-7)
        blend = weight * state['velocity'][0] + (1 - weight) * state['velocity'][1]
        direction = spectral_cascade(g[None].lerp(blend, lookahead), 'anvil2', 'bf16')
        second = .9 * second + .1 * direction.square().mean(-2, keepdim=True)
        torch.testing.assert_close(state['second'], second, rtol=1e-5, atol=2e-7)
        adapted = direction / second.sqrt().clamp_min(1e-5)
        expected = -.005 * (direction.norm() * adapted / adapted.norm()).bfloat16().float()
        torch.testing.assert_close(p - before, expected[0], atol=2e-6, rtol=2e-4)
