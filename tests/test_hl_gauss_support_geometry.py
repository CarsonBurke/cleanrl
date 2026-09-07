"""Guard against false representation wins in the support-geometry proxy."""

from dataclasses import replace

import pytest
import torch

from scripts.hlgauss.ppo_proxy_v3 import EnsembleCritic, decode, lambda_advantages
from scripts.hlgauss.support_geometry import SupportCase, initialize, resolve, sample_case


def test_rare_value_tail_preserves_each_sampled_oracle_advantage():
    case = SupportCase("tail", 3.8337, 0.3858, 0.006, 0.004224, tail_height=20.0)
    base = sample_case(replace(case, tail_height=0.0), 0, 128, 32, torch.Generator().manual_seed(1))
    shaped = sample_case(case, 0, 128, 32, torch.Generator().manual_seed(1))
    x, states, actions, rewards, truth = base
    sx, ss, sa, sr, sv = shaped
    torch.testing.assert_close(sx, x, rtol=0, atol=0)
    torch.testing.assert_close(ss, states, rtol=0, atol=0)
    torch.testing.assert_close(sa, actions, rtol=0, atol=0)
    assert float(sv.max() - truth.max()) > 10
    assert float(sv.std()) > 3 * float(truth.std())
    reference = lambda_advantages(rewards, truth[states][None], case.gamma)
    actual = lambda_advantages(sr, sv[ss][None], case.gamma)
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize("bound,center", [(10.0, 3.8), (1000.0, 30.0), (200.0, -30.0)])
def test_adaptive_geometry_does_not_get_target_informed_initial_values(bound, center):
    mapping = resolve(bound, {"target_std": 0.4, "target_median": center})
    for spec in {s.key: s for s in mapping.values()}.values():
        head = spec.build()
        model = EnsembleCritic(1, spec.bins, "sphere", True)
        initialize(model, [head])
        predictions = decode(model(torch.randn(11, 6, generator=torch.Generator().manual_seed(1))), [head])
        torch.testing.assert_close(predictions, torch.zeros_like(predictions), rtol=0, atol=bound * 2e-7)


def test_uncapped_uniform_preserves_resolution_when_budget_binds():
    mapping = resolve(1000.0, {"target_std": 0.4, "target_median": 3.8})
    capped, uncapped = mapping["capped_uniform"], mapping["uncapped_uniform"]
    assert capped.bins == 255
    assert 2 * uncapped.bound / (uncapped.bins - 1) <= 0.8
    assert 2 * uncapped.bound / (uncapped.bins - 3) > 0.8
    for spec in (capped, uncapped, mapping["asinh_bary101"]):
        head = spec.build()
        torch.testing.assert_close(head.support[[0, -1]], torch.tensor([-1000.0, 1000.0], dtype=head.support.dtype))
