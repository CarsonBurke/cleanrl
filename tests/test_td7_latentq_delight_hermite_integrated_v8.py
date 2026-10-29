"""Exact score expectations and preservation of the existing sampled DG correction."""
import ast
import inspect
import textwrap

import pytest
import torch

from cleanrl.td7.delight import td7_latentq_delight_hermite_ablation_v7 as previous
from cleanrl.td7.delight import td7_latentq_delight_hermite_integrated_v8 as trainer


def context():
    mu = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.4, -0.7], [-1.2, -0.2]], dtype=torch.float64, requires_grad=True)
    gm = torch.tensor([[2., -3.], [1., 4.]], dtype=torch.float64, requires_grad=True)
    gs = torch.tensor([[5., 2.], [-1., 3.]], dtype=torch.float64, requires_grad=True)
    return mu, log_std, gm, gs


def evaluate(eps, *, order=2, objective='delight', **gate):
    mu, log_std, gm, gs = context()
    advantage, _, _ = trainer.hermite_advantage(eps, log_std.exp(), gm, gs, order)
    u = (mu + log_std.exp() * eps).detach()
    integrated, diagnostics = trainer.integrated_hermite_score_loss(
        u, mu, log_std, advantage, gm, gs, order=order, objective=objective, **gate)
    sampled, old_diagnostics = previous.delight_score_loss(
        u, mu, log_std, advantage, objective=objective, **gate)
    sampled_pg, _ = previous.delight_score_loss(u, mu, log_std, advantage, objective='pg', **gate)
    analytic = -(mu * gm.detach()).sum(-1).mean()
    if order == 2:
        analytic = analytic - (log_std * (log_std.exp() * gs).detach()).sum(-1).mean()
    expected = analytic + sampled - sampled_pg
    return (integrated, diagnostics, expected, old_diagnostics), (mu, log_std, gm, gs)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('draw', [0.0, 0.17, 4.0])
def test_pg_gradient_exact_for_any_draws_and_detached_geometry(order, draw):
    eps = torch.full((4, 2, 2), draw, dtype=torch.float64)
    (loss, _, _, _), tensors = evaluate(eps, order=order, objective='pg')
    mu, log_std, gm, gs = tensors
    gradients = torch.autograd.grad(loss, tensors, allow_unused=True)
    torch.testing.assert_close(gradients[0], -gm.detach() / len(mu))
    expected_sigma = -log_std.detach().exp() * gs.detach() / len(mu) if order == 2 else torch.zeros_like(log_std)
    torch.testing.assert_close(gradients[1], expected_sigma)
    assert gradients[2:] == (None, None)


@pytest.mark.parametrize('scope', ['mean', 'all'])
@pytest.mark.parametrize('scale', ['rms', 'raw'])
@pytest.mark.parametrize('surprisal', ['relative', 'density'])
def test_exactly_preserves_sampled_delight_minus_pg_correction(scope, scale, surprisal):
    eps = torch.tensor([[[0.3, 1.1], [0.7, -0.2]], [[1.5, -0.4], [-0.3, 0.8]]], dtype=torch.float64)
    eps = torch.cat([eps, -eps])
    (actual, diagnostics, expected, old_diagnostics), tensors = evaluate(eps, scope=scope, scale=scale, surprisal=surprisal)
    ga = torch.autograd.grad(actual, tensors, allow_unused=True, retain_graph=True)
    ge = torch.autograd.grad(expected, tensors, allow_unused=True)
    for a, e in zip(ga[:2], ge[:2]):
        torch.testing.assert_close(a, e, atol=2e-14, rtol=2e-14)
    assert ga[2:] == ge[2:] == (None, None)
    torch.testing.assert_close(diagnostics, old_diagnostics, atol=0, rtol=0)


def test_exact_gaussian_moments_match_original_expected_dg_gradient():
    nodes = torch.tensor([3.**0.5, -3.**0.5, 0., 0., 0., 0.], dtype=torch.float64)
    eps = torch.cartesian_prod(nodes, nodes)[:, None, :].expand(-1, 2, -1)
    mu, log_std, gm, gs = context()
    advantage, _, _ = trainer.hermite_advantage(eps, log_std.exp(), gm, gs)
    u = (mu + log_std.exp() * eps).detach()
    actual, _ = trainer.integrated_hermite_score_loss(u, mu, log_std, advantage, gm, gs)
    original, _ = previous.delight_score_loss(u, mu, log_std, advantage)
    ga = torch.autograd.grad(actual, (mu, log_std), retain_graph=True)
    ge = torch.autograd.grad(original, (mu, log_std))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=2e-14, rtol=2e-14)


def test_order_one_antithetic_mean_gate_leaves_exact_pg():
    eps = torch.tensor([[[0.3, 1.1], [0.7, -0.2]], [[1.5, -0.4], [-0.3, 0.8]]], dtype=torch.float64)
    eps = torch.cat([eps, -eps])
    (actual, _, _, _), (mu, log_std, gm, _) = evaluate(eps, order=1)
    gradients = torch.autograd.grad(actual, (mu, log_std))
    torch.testing.assert_close(gradients[0], -gm.detach() / len(mu), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(gradients[1], torch.zeros_like(log_std), atol=1e-14, rtol=0)


def test_critic_training_preserved_and_new_default_is_zero_smoothing():
    for name in ('_critic_loss_fn', '_encoder_loss_fn', 'flush', 'hard_update'):
        before = ast.parse(textwrap.dedent(inspect.getsource(getattr(previous.TD7Agent, name))))
        after = ast.parse(textwrap.dedent(inspect.getsource(getattr(trainer.TD7Agent, name))))
        assert ast.dump(before) == ast.dump(after)
    assert trainer.Args().target_smoothing_sigma == 0
    assert trainer.Args().entropy_floor
    assert trainer.Args().actor_samples == 4
    assert 'clipping_statistics' in trainer.CHECKPOINT_BUFFERS


def test_clipping_statistics_accumulate_every_update_and_are_detached():
    stats = torch.zeros(5, dtype=torch.float64)
    norms = torch.tensor([0., 0.25, 1., 2.], dtype=torch.float64, requires_grad=True)
    for norm in norms:
        trainer.accumulate_clipping_statistics(stats, norm, 0.5)
    torch.testing.assert_close(stats[:3], torch.tensor([3.25, 4., 2.], dtype=torch.float64))
    torch.testing.assert_close(stats[3], (0.5 / (norms.detach() + 1e-6)).clamp(max=1).sum())
    assert stats[4].item() == 2
    assert not stats.requires_grad
