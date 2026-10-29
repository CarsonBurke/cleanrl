import pytest
import torch

from cleanrl.shared.linear_scan import reverse_linear_scan, reverse_linear_scan_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")


@pytest.mark.parametrize("steps,columns", [(1, 1), (31, 3), (32, 16), (33, 5), (1024, 16), (8192, 16)])
def test_kernel_matches_sequential_reference(steps, columns):
    generator = torch.Generator(device="cuda").manual_seed(steps * 131 + columns)
    inputs = torch.randn(steps, columns, device="cuda", generator=generator)
    coefficients = torch.rand(steps, columns, device="cuda", generator=generator) * 0.99
    coefficients[torch.rand(steps, columns, device="cuda", generator=generator) < 0.05] = 0.0
    torch.testing.assert_close(reverse_linear_scan(inputs, coefficients),
                               reverse_linear_scan_reference(inputs, coefficients), rtol=1e-5, atol=1e-5)


def test_constant_coefficient_is_discounted_sum():
    inputs = torch.ones(100, 2, device="cuda")
    outputs = reverse_linear_scan(inputs, torch.full_like(inputs, 0.5))
    expected = 2.0 * (1.0 - 0.5 ** torch.arange(100, 0, -1, device="cuda", dtype=torch.float32))
    torch.testing.assert_close(outputs[:, 0], expected)


def test_zero_coefficient_breaks_the_chain():
    inputs = torch.arange(6, dtype=torch.float32, device="cuda").view(6, 1)
    coefficients = torch.ones_like(inputs)
    coefficients[2] = 0.0
    outputs = reverse_linear_scan(inputs, coefficients)
    torch.testing.assert_close(outputs.flatten(), torch.tensor([3.0, 3.0, 2.0, 12.0, 9.0, 5.0], device="cuda"))


def test_non_cuda_falls_back_to_reference():
    inputs, coefficients = torch.randn(7, 2), torch.rand(7, 2)
    torch.testing.assert_close(reverse_linear_scan(inputs, coefficients),
                               reverse_linear_scan_reference(inputs, coefficients))
