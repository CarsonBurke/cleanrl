"""Analytical scalar-signal contracts; execute CUDA cases through mlq."""

import pytest
import torch

from cleanrl.plasticity.ppo_signal_legibility import anova

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _statistics(cells):
    """Reduce explicit (bin, sample, unit) observations, not invented moments."""
    return anova(cells.sum(1), cells.square().sum(1), cells.shape[1], cells.shape[0])


def test_nonnegative_squared_snr_gate_is_invariant_to_signal_sign():
    means = torch.tensor([1.0, 2.0, -4.0, -3.0], device="cuda", dtype=torch.float64)
    residual = means.new_tensor([-1.0, 1.0])
    cells = (means[:, None] + residual[None, :]).unsqueeze(-1)

    observed = _statistics(cells)
    reversed_signal = _statistics(-cells)

    # A global sign change cannot affect a squared objective over the same
    # nonnegative weights. The stronger half is negative before the reversal.
    for key in ("gate_abs", "gate_gain"):
        torch.testing.assert_close(observed[key], reversed_signal[key])


def test_squared_snr_gate_selects_the_larger_same_sign_energy():
    means = torch.tensor([[1.0, 1.0, -1.0], [2.0, 2.0, -2.0],
                          [-4.0, 3.0, -3.0], [-3.0, 4.0, -4.0]],
                         device="cuda", dtype=torch.float64)
    residual = means.new_tensor([-1.0, 1.0])
    cells = means[:, None, :] + residual[None, :, None]

    observed = _statistics(cells)

    # Each cell's population SD is one. Cauchy-Schwarz on each sign half
    # gives max(1^2+2^2, 4^2+3^2)/4 = 6.25 for the mixed-sign unit.
    # Either all-same-sign unit attains the signed reference 30/4 = 7.5.
    expected_gate = means.new_tensor([6.25, 7.5, 7.5])
    expected_uniform = means.new_tensor([1.0, 6.25, 6.25])
    expected_signed = means.new_full((3,), 7.5)
    torch.testing.assert_close(observed["gate_abs"], expected_gate)
    torch.testing.assert_close(observed["uniform"], expected_uniform)
    torch.testing.assert_close(observed["attainable"], expected_signed)
    torch.testing.assert_close(observed["gate_gain"], expected_gate / expected_uniform)
    torch.testing.assert_close(observed["oracle_gain"], expected_signed / expected_uniform)


def test_standardized_anova_uses_residual_degrees_and_total_energy():
    means = torch.tensor([[0.0, -10.0], [2.0, 10.0]], device="cuda", dtype=torch.float64)
    residual = means.new_tensor([-1.0, 1.0])
    cells = means[:, None, :] + residual[None, :, None]
    # Different positive bin-constant gains must cancel upon standardization.
    cells = cells * means.new_tensor([2.0, 0.5])[:, None, None]

    observed = _statistics(cells)

    # Standardized residual SS = 4, residual df = 2, MSW = 2, not 1.
    # Between-bin SS = [4, 400], hence F = [2, 200]. The adjusted effect
    # denominator is total SS = [8, 404], not sample count = 4.
    torch.testing.assert_close(observed["f_snr"], means.new_tensor([2.0, 200.0]))
    torch.testing.assert_close(observed["eta_snr"], means.new_tensor([0.25, 398.0 / 404.0]))
