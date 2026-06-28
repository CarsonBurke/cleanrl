import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_v162critic_d3hlgauss_mtp_v2 import (
    Dreamer3HLGaussSupport,
)
from cleanrl.shared.hl_gauss import symexp


def test_dreamer3_hlgauss_support_matches_symexp_half_mirror_bins():
    support = Dreamer3HLGaussSupport(511, -20.0, 20.0, 2.0, torch.device("cpu"))
    half = torch.linspace(-20.0, 0.0, 256)
    expected = torch.cat([symexp(half), -symexp(half[:-1]).flip(0)], dim=0)

    assert torch.allclose(support.support, expected)
    assert support.support[255].item() == 0.0
    assert torch.allclose(support.support, -support.support.flip(0))


def test_dreamer3_hlgauss_projection_normalizes_and_peaks_at_center():
    support = Dreamer3HLGaussSupport(511, -20.0, 20.0, 2.0, torch.device("cpu"))
    targets = torch.tensor([-10000.0, 0.0, 10000.0])
    probs = support.project(targets)

    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-6)
    assert probs[1].argmax().item() == 255


def test_dreamer3_hlgauss_decode_is_raw_expected_value_not_symexp_mean_coord():
    support = Dreamer3HLGaussSupport(511, -20.0, 20.0, 2.0, torch.device("cpu"))
    logits = torch.full((1, 511), -100.0)
    logits[0, 300] = 0.0
    logits[0, 360] = 1.0
    probs = torch.softmax(logits, dim=-1)
    coord_decode = symexp((probs * support.coord_support).sum(-1))

    assert torch.allclose(support.to_expected_scalar(logits), (probs * support.support).sum(-1))
    assert not torch.allclose(support.to_expected_scalar(logits), coord_decode)
    assert support.to_expected_scalar(torch.zeros(1, 511)).abs().item() < 1e-6
