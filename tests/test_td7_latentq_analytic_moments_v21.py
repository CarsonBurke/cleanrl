"""Pure tensor math checks; no networks, simulator, CUDA execution or smoke runs."""
import math

import numpy as np
import pytest
import torch

from cleanrl.td7.latentq.td7_latentq_analytic_moments_v21 import (
    analytic_basis_value,
    clipped_gaussian_hinge,
    clipped_gaussian_moments,
)

DTYPE = torch.float64
KNOTS = torch.linspace(-2 / 3, 2 / 3, 5, dtype=DTYPE)


def quadrature(mu, sigma, function, low=-1., high=1.):
    """Independent physical-action density integral plus both saturation atoms."""
    mu, sigma = float(mu), float(sigma)
    x, w = np.polynomial.legendre.leggauss(256)
    x = (high + low) / 2 + x * (high - low) / 2
    w = w * (high - low) / 2
    density = np.exp(-.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    cdf = lambda z: .5 * math.erfc(-z / math.sqrt(2))
    return (w * density * function(x)).sum() + cdf((low - mu) / sigma) * function(low) + cdf((mu - high) / sigma) * function(high)


@pytest.mark.parametrize('mu,sigma', [(0., .1), (.92, .24), (-1.3, .4), (2., 1.5), (0., 3.)])
def test_clipped_moments_include_saturation_atoms(mu, sigma):
    m1, m2 = clipped_gaussian_moments(torch.tensor([[mu]], dtype=DTYPE), torch.tensor([[sigma]], dtype=DTYPE))
    assert m1.item() == pytest.approx(quadrature(mu, sigma, lambda x: x), abs=2e-12)
    assert m2.item() == pytest.approx(quadrature(mu, sigma, lambda x: x*x), abs=2e-12)
    assert m2 >= m1.square() - 1e-13
    assert m2 <= 1 + 1e-13


def test_nonstandard_bounds():
    mu, sigma = torch.tensor([[1.4]], dtype=DTYPE), torch.tensor([[.7]], dtype=DTYPE)
    m1, m2 = clipped_gaussian_moments(mu, sigma, low=-2., high=.5)
    for actual, power in [(m1, 1), (m2, 2)]:
        assert actual.item() == pytest.approx(quadrature(1.4, .7, lambda x: x**power, -2., .5), abs=2e-12)


@pytest.mark.parametrize('mu,sigma', [(-1.4, .3), (.8, .4), (0., 2.)])
def test_hinge_expectations_against_split_density_integral(mu, sigma):
    actual = clipped_gaussian_hinge(torch.tensor([[mu]], dtype=DTYPE), torch.tensor([[sigma]], dtype=DTYPE), KNOTS)
    # Integrate only from each kink to upper bound to avoid quadrature kink error.
    for value, knot in zip(actual.flatten(), KNOTS):
        k = knot.item()
        x, w = np.polynomial.legendre.leggauss(128)
        x, w = k + (x+1)*(1-k)/2, w*(1-k)/2
        pdf = np.exp(-.5*((x-mu)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        upper_atom = .5*math.erfc((1-mu)/(sigma*math.sqrt(2)))
        expected = (w*pdf*(x-k)).sum() + upper_atom*(1-k)
        assert value.item() == pytest.approx(expected, abs=2e-12)


def test_zero_sigma_point_limit_and_unused_branch_derivatives():
    mu = torch.tensor([[-2., -2/3, -.2, 0., 2/3, 2.]], dtype=DTYPE, requires_grad=True)
    sigma = torch.zeros_like(mu, requires_grad=True)
    m1, m2 = clipped_gaussian_moments(mu, sigma)
    hinge = clipped_gaussian_hinge(mu, sigma, KNOTS)
    a = mu.clamp(-1, 1)
    torch.testing.assert_close(m1, a)
    torch.testing.assert_close(m2, a.square())
    torch.testing.assert_close(hinge, (a.unsqueeze(-1)-KNOTS).relu())
    grads = torch.autograd.grad(m1.sum()+m2.sum()+hinge.sum(), (mu, sigma))
    assert all(torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(grads[1], torch.zeros_like(sigma))


def test_full_cross_interaction_and_diagonal_convention():
    indices = torch.triu_indices(2, 2)
    # V,g0,g1,H00,H01,H11,hinges: H01 appears once, diag gets .5.
    coeff = torch.zeros(1, 2, 1+2+3+10, dtype=DTYPE)
    coeff[:, :, :6] = torch.tensor([2., 3., -4., 5., 7., -6.], dtype=DTYPE)
    mu, sigma = torch.tensor([[.7, -.8]], dtype=DTYPE), torch.tensor([[.4, .5]], dtype=DTYPE)
    actual = analytic_basis_value(coeff, mu, sigma, KNOTS, indices)
    m1 = [quadrature(m, s, lambda x: x) for m, s in zip(mu[0], sigma[0])]
    m2 = [quadrature(m, s, lambda x: x*x) for m, s in zip(mu[0], sigma[0])]
    expected = 2+3*m1[0]-4*m1[1]+2.5*m2[0]+7*m1[0]*m1[1]-3*m2[1]
    torch.testing.assert_close(actual, torch.full_like(actual, expected))
    point = analytic_basis_value(coeff, mu, torch.zeros_like(sigma), KNOTS, indices)
    expected_point = 2+3*.7-4*(-.8)+2.5*.7**2+7*.7*(-.8)-3*.8**2
    torch.testing.assert_close(point, torch.full_like(point, expected_point))


def test_analytic_value_query_and_coefficient_gradients():
    generator = torch.Generator().manual_seed(1)
    indices = torch.triu_indices(2, 2)
    coeff = torch.randn(1, 2, 16, generator=generator, dtype=DTYPE, requires_grad=True)
    mu = torch.tensor([[.93, -1.1]], dtype=DTYPE, requires_grad=True)
    sigma = torch.tensor([[.3, .4]], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c,m,s: analytic_basis_value(c,m,s,KNOTS,indices), (coeff,mu,sigma), eps=1e-6, atol=2e-6, rtol=2e-5)


def test_mean_and_scale_derivatives_against_independent_quadrature():
    mu = torch.tensor([[.85]], dtype=DTYPE, requires_grad=True)
    sigma = torch.tensor([[.35]], dtype=DTYPE, requires_grad=True)
    m1, m2 = clipped_gaussian_moments(mu, sigma)
    actual = torch.autograd.grad((2*m1+3*m2).sum(), (mu,sigma))
    h=1e-5
    fn=lambda m,s: quadrature(m,s,lambda x: 2*x+3*x*x)
    expected=[(fn(.85+h,.35)-fn(.85-h,.35))/(2*h), (fn(.85,.35+h)-fn(.85,.35-h))/(2*h)]
    for g, e in zip(actual,expected):
        assert g.item() == pytest.approx(e, abs=1e-8)


def test_exact_lap_compensation_with_excluded_lane():
    # Expected sampled objective equals uniform fitting-row MSE, without rescaling.
    p=torch.tensor([.1,.2,.3,.4],dtype=DTYPE)
    fit=torch.tensor([0.,1.,0.,1.],dtype=DTYPE)
    residual=torch.tensor([7.,2.,9.,3.],dtype=DTYPE)
    weighted=residual.square()*fit/(fit.sum()*p)
    torch.testing.assert_close((p*weighted).sum(), residual[[1,3]].square().mean())
