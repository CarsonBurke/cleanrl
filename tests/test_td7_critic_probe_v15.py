"""Validate diagnostic math without constructing or running an ML model."""
import pytest
import torch

from cleanrl.td7.latentq.td7_latentq_critic_probe_v15 import gaussian_kl, probe_change


def snapshot(gradient, center=0.0, point=0.0, label=0.0):
    return {
        "gradient": torch.tensor(gradient, dtype=torch.float64),
        "center": torch.full((2, 2), center, dtype=torch.float64),
        "point": torch.full((2, 2), point, dtype=torch.float64),
        "label": torch.full((2, 1), label, dtype=torch.float64),
    }


def test_value_offset_does_not_imply_gradient_drift():
    before = snapshot([[1.0, 2.0], [3.0, 4.0]])
    after = snapshot([[1.0, 2.0], [3.0, 4.0]], center=7.0, point=5.0, label=2.0)
    result = probe_change(before, after)
    assert result["center_rms_change"] == 7.0
    assert result["point_rms_change"] == 5.0
    assert result["label_rms_change"] == 2.0
    assert result["gradient_rms_change"] == 0.0
    assert result["gradient_cosine"] == pytest.approx(1.0)
    assert result["gradient_norm_ratio"] == 1.0


def test_gradient_flip_and_rescaling_are_distinguished():
    result = probe_change(snapshot([[3.0, 4.0]]), snapshot([[-6.0, -8.0]]))
    assert result["gradient_cosine"] == pytest.approx(-1.0)
    assert result["gradient_norm_ratio"] == 2.0
    assert result["gradient_rms_change"] == pytest.approx((225.0 / 2) ** 0.5)


@pytest.mark.parametrize("before,after", [([0.0, 0.0], [1.0, 0.0]), ([0.0, 0.0], [0.0, 0.0])])
def test_zero_reference_has_undefined_direction_and_relative_size(before, after):
    result = probe_change(snapshot([before]), snapshot([after]))
    assert torch.isnan(result["gradient_cosine"])
    assert torch.isnan(result["gradient_norm_ratio"])
    assert torch.isfinite(result["gradient_rms_change"])


def test_gradient_collapse_is_zero_ratio_with_undefined_direction():
    result = probe_change(snapshot([[1.0, 0.0]]), snapshot([[0.0, 0.0]]))
    assert result["gradient_norm_ratio"] == 0.0
    assert torch.isnan(result["gradient_cosine"])


def test_behavior_kl_uses_full_gaussian_and_sums_action_dimensions():
    mu_old = torch.tensor([[0.0, 0.0]])
    sigma_old = torch.tensor([[2.0, 2.0]])
    mu_new = torch.tensor([[1.0, 1.0]])
    log_std_new = torch.zeros_like(mu_new)
    mean, scale = gaussian_kl(mu_old, sigma_old, mu_new, log_std_new)
    assert mean.item() == pytest.approx(1.0)
    assert scale.item() == pytest.approx(3.0 - 2.0 * torch.log(torch.tensor(2.0)).item())
