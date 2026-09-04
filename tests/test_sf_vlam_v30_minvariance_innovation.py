import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v30_minvariance_innovation.py"
    spec = importlib.util.spec_from_file_location("sf_vlam_v30_minvariance_innovation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _fit(regressor, psi, trace, targets, epochs=60, lr=3e-3, seed=0):
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
    return MODULE.fit_innovation_regressor(
        regressor,
        optimizer,
        psi,
        trace,
        targets,
        epochs,
        512,
        10.0,
        torch.Generator().manual_seed(seed),
    )


def test_regressor_starts_as_a_pure_state_baseline():
    """u's head is std=0.01-initialized: at iteration 1 the credit is a rounding error.

    A regressor that started with a large random u would drive the actor with noise
    before it had learned anything -- the failure the warm-up exists to prevent, and
    this init is the second line of defence against it.
    """
    torch.manual_seed(0)
    regressor = MODULE.InnovationAdvantageRegressor(8, 32)
    psi = torch.randn(256, 8)
    covector = regressor.covector(psi)
    assert float(covector.norm(dim=-1).mean()) < 0.2


def test_covector_is_consistent_between_the_two_entry_points():
    torch.manual_seed(1)
    regressor = MODULE.InnovationAdvantageRegressor(8, 32)
    psi = torch.randn(64, 8)
    _, covector = regressor(psi)
    torch.testing.assert_close(covector, regressor.covector(psi))


def test_recovers_the_innovation_covector_of_a_partially_linear_return():
    """G = b(psi) + w.E with a NONLINEAR state baseline.

    The point of the partially-linear fit is that b absorbs the state so u is left with
    the action's contribution. If the baseline were linear this would pass trivially; the
    tanh baseline is what makes it a real test of the partialling-out.
    """
    torch.manual_seed(2)
    dim = 6
    weight = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5, -0.75])
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim)
    baseline = 3.0 * torch.tanh(psi[:, 0]) - psi[:, 1].square()
    targets = baseline + trace @ weight

    regressor = MODULE.InnovationAdvantageRegressor(dim, 64)
    _fit(regressor, psi, trace, targets)
    covector = regressor.covector(psi)
    cosine = torch.nn.functional.cosine_similarity(covector.mean(0), weight, dim=0)
    assert float(cosine) > 0.98, float(cosine)
    torch.testing.assert_close(covector.mean(0), weight, atol=0.2, rtol=0.2)


def test_innovation_term_is_shrunk_to_zero_when_the_innovation_is_noise():
    """The whole claim of minimum-variance weighting.

    When the innovation explains nothing, the least-squares covector goes to zero rather
    than to some arbitrary unit direction -- so the credit vanishes instead of injecting
    noise into the policy. This is exactly what a unit-norm ordinal covector cannot do.
    """
    torch.manual_seed(3)
    dim = 6
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim)
    targets = 3.0 * torch.tanh(psi[:, 0]) - psi[:, 1].square()  # no innovation term

    regressor = MODULE.InnovationAdvantageRegressor(dim, 64)
    _fit(regressor, psi, trace, targets)
    with torch.no_grad():
        baseline, covector = regressor(psi)
        credit = (covector * trace).sum(-1)
    assert float(credit.square().mean().sqrt()) < 0.15 * float(
        targets.square().mean().sqrt()
    )
    baseline_ev = MODULE.explained_variance_1d(baseline, targets)
    full_ev = MODULE.explained_variance_1d(baseline + credit, targets)
    assert full_ev - baseline_ev < 0.05


def test_innovation_delta_explained_variance_is_positive_when_it_should_be():
    """The design's declared falsifier must actually fire in both directions."""
    torch.manual_seed(4)
    dim = 6
    weight = torch.tensor([1.0, -1.0, 0.0, 2.0, 0.5, 0.0])
    psi = torch.randn(4096, dim)
    trace = torch.randn(4096, dim)
    targets = torch.tanh(psi[:, 0]) + trace @ weight

    regressor = MODULE.InnovationAdvantageRegressor(dim, 64)
    _fit(regressor, psi, trace, targets)
    with torch.no_grad():
        baseline, covector = regressor(psi)
    baseline_ev = MODULE.explained_variance_1d(baseline, targets)
    full_ev = MODULE.explained_variance_1d(
        baseline + (covector * trace).sum(-1), targets
    )
    assert full_ev - baseline_ev > 0.5, (baseline_ev, full_ev)


def test_fit_is_a_no_op_on_an_empty_gate():
    regressor = MODULE.InnovationAdvantageRegressor(6, 16)
    before = [p.clone() for p in regressor.parameters()]
    loss, grad_norm, steps = _fit(
        regressor, torch.zeros(0, 6), torch.zeros(0, 6), torch.zeros(0), epochs=5
    )
    assert steps == 0 and loss != loss and grad_norm != grad_norm  # NaN, not a crash
    for parameter, original in zip(regressor.parameters(), before):
        torch.testing.assert_close(parameter, original)


def test_return_residual_correlation_uses_the_cross_env_baseline():
    torch.manual_seed(5)
    returns = torch.randn(20, 8)
    residual = returns - returns.mean(1, keepdim=True)
    valid = torch.ones(20, 8, dtype=torch.bool)
    assert abs(MODULE.credit_return_correlation(residual, returns, valid) - 1.0) < 1e-5
    shifted = returns + torch.arange(20.0).unsqueeze(1)
    assert abs(MODULE.credit_return_correlation(residual, shifted, valid) - 1.0) < 1e-5
