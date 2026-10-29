"""Prefix-JVP tensor identities; no networks, environments, CUDA or training."""

import pytest
import torch
import torch.nn.functional as F

from cleanrl.td7.delight.td7_latentq_delight_fast_preaction_v3 import (
    avg_l1_norm_jvp,
    elu_jvp,
    sample_blocked_priorities,
    update_priority_blocks,
)


@pytest.mark.parametrize("tangent_batch", [3, 9])
def test_normalization_prefix_jvp_matches_full_zero_padded_tangent_and_gradients(tangent_batch):
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(9, 2, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    t = torch.randn(tangent_batch, 2, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    primal, tangent = avg_l1_norm_jvp(x, t)
    full_t = F.pad(t, (0, 0, 0, 0, 0, 9 - tangent_batch))
    scale = x.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    reference_primal = x / scale
    reference_tangent = (full_t - reference_primal * (x.sign() * full_t).mean(-1, keepdim=True)) / scale
    torch.testing.assert_close(primal, reference_primal)
    torch.testing.assert_close(tangent, reference_tangent[:tangent_batch])
    actual = torch.autograd.grad(primal.square().sum() + tangent.square().sum(), (x, t), retain_graph=True)
    expected = torch.autograd.grad(reference_primal.square().sum() + reference_tangent.square().sum(), (x, t))
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_blocked_sampling_has_exact_priority_law_and_excludes_padding():
    priority = torch.tensor([1., 3., 0., 2., 4., 0., 0., 0.], dtype=torch.float64)
    masses = priority.view(-1, 2).sum(1)
    grid = (torch.arange(100, dtype=torch.float64) + .5) / 100
    uniforms = torch.cartesian_prod(grid, grid)
    indices = sample_blocked_priorities(priority, masses, uniforms, 2)
    frequencies = torch.bincount(indices, minlength=8) / len(indices)
    torch.testing.assert_close(frequencies.to(torch.float64), priority / priority.sum())


def test_zero_uniforms_skip_empty_prefixes():
    priority = torch.tensor([0., 0., 0., 5., 0., 0.])
    masses = priority.view(-1, 2).sum(1)
    indices = sample_blocked_priorities(priority, masses, torch.zeros(3, 2), 2)
    assert indices.tolist() == [3, 3, 3]


def test_duplicate_priority_updates_refresh_blocks_without_double_counting():
    priority = torch.tensor([1., 3., 0., 2., 4., 0., 0., 0.])
    masses = priority.view(-1, 2).sum(1)
    indices = torch.tensor([0, 0, 3, 4])
    values = torch.tensor([5., 5., 7., 0.])
    update_priority_blocks(priority, masses, indices, values, 2)
    torch.testing.assert_close(priority, torch.tensor([5., 3., 0., 7., 0., 0., 0., 0.]))
    torch.testing.assert_close(masses, priority.view(-1, 2).sum(1))


@pytest.mark.parametrize("shape,tangent_batch", [((9, 5), 3), ((2, 9, 5), 3), ((2, 9, 5), 9)])
def test_elu_prefix_jvp_matches_full_zero_padded_tangent_and_gradients(shape, tangent_batch):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    tangent_shape = (*shape[:-2], tangent_batch, shape[-1])
    t = torch.randn(tangent_shape, generator=generator, dtype=torch.float64, requires_grad=True)
    primal, tangent = elu_jvp(x, t)
    full_t = F.pad(t, (0, 0, 0, shape[-2] - tangent_batch))
    reference_primal = F.elu(x)
    reference_tangent = torch.where(x > 0, full_t, full_t * x.clamp(max=0).exp())
    torch.testing.assert_close(primal, reference_primal)
    torch.testing.assert_close(tangent, reference_tangent[..., :tangent_batch, :])
    actual = torch.autograd.grad(primal.square().sum() + tangent.square().sum(), (x, t), retain_graph=True)
    expected = torch.autograd.grad(reference_primal.square().sum() + reference_tangent.square().sum(), (x, t))
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)
