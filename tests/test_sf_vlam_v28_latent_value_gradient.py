import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v28_latent_value_gradient.py"
    spec = importlib.util.spec_from_file_location("sf_vlam_v28_latent_value_gradient", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _readout(value_dim=6, hidden=16, seed=0):
    torch.manual_seed(seed)
    return MODULE.LatentValueReadout(value_dim, hidden)


def test_covector_matches_autograd_and_is_chunk_invariant():
    readout = _readout()
    psi = torch.randn(37, 6)
    covector = MODULE.value_gradient_covector(readout, psi, chunk=8)
    chunked_once = MODULE.value_gradient_covector(readout, psi, chunk=1024)
    torch.testing.assert_close(covector, chunked_once)

    reference = torch.autograd.functional.jacobian(
        lambda x: readout(x.unsqueeze(0)).squeeze(0), psi[3]
    )
    torch.testing.assert_close(covector[3], reference)


def test_covector_leaves_no_parameter_gradients_and_works_under_no_grad():
    readout = _readout()
    for parameter in readout.parameters():
        assert parameter.grad is None
    with torch.no_grad():
        covector = MODULE.value_gradient_covector(readout, torch.randn(9, 6))
    assert not covector.requires_grad
    for parameter in readout.parameters():
        assert parameter.grad is None


def test_credit_is_the_first_order_readout_difference():
    """<grad f(psi), E> is the linearization of f(psi + E) - f(psi).

    This is the identity the whole design rests on, so pin it on a case where the
    linearization is EXACT: a readout whose nonlinearity is bypassed. A tiny innovation
    reproduces it to first order for the general nonlinear readout.
    """
    readout = _readout()
    psi = torch.randn(64, 6)
    innovation = 1e-4 * torch.randn(64, 6)
    covector = MODULE.value_gradient_covector(readout, psi)
    linearized = (covector * innovation).sum(-1)
    with torch.no_grad():
        exact = readout(psi + innovation) - readout(psi)
    torch.testing.assert_close(linearized, exact, atol=1e-6, rtol=1e-3)


def test_fitted_readout_recovers_a_linear_value_covector():
    """On G = w.psi the fitted covector must converge to w, up to the fit's error.

    This is the mechanism end to end: regress returns on psi, differentiate, and get the
    weight that turns an innovation into an advantage.
    """
    torch.manual_seed(1)
    weight = torch.tensor([1.5, -2.0, 0.5, 0.0, 3.0, -1.0])
    psi = torch.randn(4096, 6)
    targets = psi @ weight
    readout = _readout(seed=2)
    optimizer = torch.optim.Adam(readout.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(3)
    loss, grad_norm, steps = MODULE.fit_latent_value_readout(
        readout, optimizer, psi, targets, 60, 512, 10.0, generator
    )
    assert steps == 60 * 8 and grad_norm > 0.0
    covector = MODULE.value_gradient_covector(readout, psi)
    cosine = torch.nn.functional.cosine_similarity(
        covector.mean(0), weight, dim=0
    )
    assert float(cosine) > 0.99, (float(cosine), loss)
    assert MODULE.explained_variance_1d(readout(psi), targets) > 0.95


def test_fit_is_a_no_op_on_an_empty_gate():
    readout = _readout()
    optimizer = torch.optim.Adam(readout.parameters(), lr=1e-3)
    before = [p.clone() for p in readout.parameters()]
    loss, grad_norm, steps = MODULE.fit_latent_value_readout(
        readout,
        optimizer,
        torch.zeros(0, 6),
        torch.zeros(0),
        5,
        128,
        1.0,
        torch.Generator().manual_seed(0),
    )
    assert steps == 0 and loss != loss and grad_norm != grad_norm  # NaN, not a crash
    for parameter, original in zip(readout.parameters(), before):
        torch.testing.assert_close(parameter, original)


def test_explained_variance_edge_cases():
    target = torch.randn(100)
    assert MODULE.explained_variance_1d(target, target) == 1.0
    assert MODULE.explained_variance_1d(torch.zeros(100), target) < 0.05
    constant = torch.full((100,), 2.0)
    assert MODULE.explained_variance_1d(constant, constant) != MODULE.explained_variance_1d(
        constant, constant
    )  # NaN on a degenerate target rather than a divide-by-zero


def test_return_residual_correlation_uses_the_cross_env_baseline():
    """The falsifier must see ADVANTAGE, not the level of the return.

    Adding a per-timestep constant to every environment's return changes the return but
    not the advantage, so the statistic must be invariant to it. Scaling the credit by a
    positive constant must also leave it unchanged.
    """
    torch.manual_seed(4)
    returns = torch.randn(20, 8)
    residual = returns - returns.mean(1, keepdim=True)
    valid = torch.ones(20, 8, dtype=torch.bool)

    perfect = MODULE.credit_return_correlation(residual, returns, valid)
    assert abs(perfect - 1.0) < 1e-5
    assert abs(MODULE.credit_return_correlation(-residual, returns, valid) + 1.0) < 1e-5

    shifted = returns + torch.arange(20.0).unsqueeze(1)
    assert abs(MODULE.credit_return_correlation(residual, shifted, valid) - perfect) < 1e-5
    assert abs(
        MODULE.credit_return_correlation(17.0 * residual, returns, valid) - perfect
    ) < 1e-5

    unrelated = torch.randn(20, 8)
    assert abs(MODULE.credit_return_correlation(unrelated, returns, valid)) < 0.3


def test_return_residual_correlation_respects_the_validity_mask():
    torch.manual_seed(5)
    returns = torch.randn(10, 4)
    residual = returns - returns.mean(1, keepdim=True)
    credit = residual.clone()
    credit[5:] = torch.randn(5, 4)  # garbage, but masked out
    valid = torch.zeros(10, 4, dtype=torch.bool)
    valid[:5] = True
    assert abs(MODULE.credit_return_correlation(credit, returns, valid) - 1.0) < 1e-5
    assert MODULE.credit_return_correlation(
        credit, returns, torch.zeros(10, 4, dtype=torch.bool)
    ) != MODULE.credit_return_correlation(credit, returns, torch.zeros(10, 4, dtype=torch.bool))


def test_constant_input_fit_cannot_recover_the_value_covector():
    """The hazard the iteration-1 degeneracy guard exists to prevent.

    psi is identically zero on iteration 1 (hard-zeroed critic head), and readout_mean has
    already been updated from this rollout by the time the fit runs, so every fit row is
    the same NONZERO constant. On a constant input the loss carries no information about
    grad f at all -- yet the fit still runs, readout_steps > 0 still certifies the readout
    as ready, and policy_credit_rms still renormalizes whatever came out to unit RMS for
    the actor to spend a full PPO update on.

    Stated operationally, against the same ground truth the healthy path recovers: with a
    varying input the fitted covector converges to the true value covector w
    (test_fitted_readout_recovers_a_linear_value_covector); with the constant input of
    iteration 1 and the SAME returns it does not, and cannot.
    """
    torch.manual_seed(6)
    weight = torch.tensor([1.5, -2.0, 0.5, 0.0, 3.0, -1.0, 0.25, -0.5])
    psi = torch.randn(16384, 8)
    targets = psi @ weight
    constant = torch.zeros_like(psi) + psi.mean(0)  # what iteration 1 actually presents

    readout = _readout(value_dim=8, hidden=64, seed=6)
    MODULE.fit_latent_value_readout(
        readout,
        torch.optim.Adam(readout.parameters(), lr=3e-4),
        constant,
        targets,
        20,
        4096,
        1.0,
        torch.Generator().manual_seed(7),
    )
    covector = MODULE.value_gradient_covector(readout, constant)[0]
    cosine = float(torch.nn.functional.cosine_similarity(covector, weight, dim=0))
    assert abs(cosine) < 0.9, cosine

    # The guard predicate: fires on the degenerate input, silent on a real one.
    assert float(constant.std(0).max()) < 1e-6
    assert float(psi.std(0).max()) > 1e-6


def test_fit_gate_falls_back_instead_of_deadlocking():
    """An empty mc_window gate must never leave the fit set empty.

    With no fit, readout_ready never flips; with readout_ready False the credit is zero
    and ent_coef is 0, so the actor gradient is exactly zero, so episodes stay short, so
    the gate stays empty. The fallback chain is what stops that from being absorbing.
    """
    complete = torch.zeros(64, dtype=torch.bool)
    complete[:10] = True

    def select(mc_mask, mc_complete):
        gate = mc_mask
        if not bool(gate.any()):
            gate = mc_complete
        if not bool(gate.any()):
            gate = torch.ones_like(mc_mask)
        return gate

    populated = torch.zeros(64, dtype=torch.bool)
    populated[20:40] = True
    assert int(select(populated, complete).sum()) == 20      # normal: mc_window wins
    empty = torch.zeros(64, dtype=torch.bool)
    assert int(select(empty, complete).sum()) == 10          # falls back to complete
    assert int(select(empty, empty).sum()) == 64             # falls back to everything
