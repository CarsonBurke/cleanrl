import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v27_ordinal_vector_moment.py"
    spec = importlib.util.spec_from_file_location("sf_vlam_v27_ordinal_vector_moment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_ordinal_targets_use_only_order_not_return_magnitude():
    traces = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]]
    )
    returns = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    horizons = torch.full((2, 3), 100.0)
    direction, group_valid, count, agreement = MODULE.ordinal_vector_targets(
        traces, returns, valid, horizons, 0
    )
    scaled, scaled_valid, scaled_count, scaled_agreement = MODULE.ordinal_vector_targets(
        traces, 17.0 + 1000.0 * returns, valid, horizons, 0
    )
    torch.testing.assert_close(scaled, direction)
    torch.testing.assert_close(scaled_valid, group_valid)
    torch.testing.assert_close(scaled_count, count)
    torch.testing.assert_close(scaled_agreement, agreement)
    assert group_valid.all() and count.tolist() == [3, 3]
    torch.testing.assert_close(direction.norm(dim=-1), torch.ones(2))


def test_ordinal_targets_respect_completion_and_horizon_masks():
    traces = torch.randn(1, 4, 5)
    returns = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    complete = torch.tensor([[True, True, False, True]])
    horizons = torch.tensor([[100.0, 101.0, 100.0, 130.0]])
    _, valid, count, _ = MODULE.ordinal_vector_targets(
        traces, returns, complete, horizons, 2
    )
    assert valid.item()
    assert count.item() == 1


def test_grouped_moment_is_exactly_zero_at_behavior_and_scale_equivariant():
    torch.manual_seed(1)
    traces = torch.randn(7, 8, 11)
    ratio = torch.ones(7, 8)
    baseline = torch.randn(7, 11)
    zero = MODULE.grouped_vector_moment(ratio, traces, baseline, torch.tensor(2.0))
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    perturbation = 0.1 * torch.randn_like(ratio)
    shifted = MODULE.grouped_vector_moment(
        ratio + perturbation, traces, baseline, torch.tensor(2.0)
    )
    scaled = MODULE.grouped_vector_moment(
        ratio + perturbation, 9.0 * traces, 9.0 * baseline, torch.tensor(18.0)
    )
    torch.testing.assert_close(scaled, shifted)


def test_grouped_moment_gradient_moves_toward_full_vector_target():
    torch.manual_seed(2)
    time, envs, dim = 12, 10, 7
    traces = torch.randn(time, envs, dim)
    target = torch.nn.functional.normalize(torch.randn(time, dim), dim=-1)
    baseline = torch.randn(time, dim)
    logits = torch.zeros(time, envs, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.08)
    initial = None
    for _ in range(120):
        moment = MODULE.grouped_vector_moment(
            logits.exp(), traces, baseline, torch.tensor(1.0)
        )
        loss = 0.5 * (moment - target).square().sum(-1).mean()
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.35 * initial


def test_analytic_beta_raw_score_matches_autograd():
    torch.manual_seed(3)
    action_dim = 3
    raw = torch.randn(9, 2 * action_dim, requires_grad=True)
    actions = MODULE.beta_from_raw(raw.detach(), action_dim).sample()
    log_prob = MODULE.beta_from_raw(raw, action_dim).log_prob(actions).sum()
    expected = torch.autograd.grad(log_prob, raw)[0]
    actual = MODULE.beta_raw_score(raw.detach(), actions, action_dim)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_exact_beta_kl_is_zero_only_at_identity():
    torch.manual_seed(4)
    raw = torch.randn(64, 12)
    torch.testing.assert_close(
        MODULE.mean_beta_kl(raw, raw, 6), torch.tensor(0.0), atol=1e-7, rtol=0
    )
    assert MODULE.mean_beta_kl(raw, raw + 0.2 * torch.randn_like(raw), 6) > 0
