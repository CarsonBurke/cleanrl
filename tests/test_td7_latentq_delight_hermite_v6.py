"""Native latent-value score identities on tensors, without networks or GPU work."""
import ast
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from cleanrl.td7.delight import td7_latentq_delight_hermite_v6 as trainer


@pytest.fixture(autouse=True, params=["td7_latentq_delight_hermite_v6", "td7_latentq_delight_hermite_ablation_v7", "td7_latentq_delight_hermite_integrated_v8"])
def native_implementation(request, monkeypatch):
    module = importlib.import_module(f"cleanrl.td7.delight.{request.param}")
    monkeypatch.setitem(globals(), "trainer", module)


def gaussian_fourth_moment_rule():
    # Equal-weight replication of the Gaussian 3-node rule, exact through fifth degree.
    one_dim = torch.tensor([3.0**0.5, -(3.0**0.5), 0, 0, 0, 0], dtype=torch.float64)
    return torch.cartesian_prod(one_dim, one_dim)[:, None, :].expand(-1, 2, -1)


def test_order_two_pg_recovers_both_native_value_gradients():
    mu = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.4, -0.7], [-1.2, -0.2]], dtype=torch.float64, requires_grad=True)
    grad_mu = torch.tensor([[2.0, -3.0], [1.0, 4.0]], dtype=torch.float64, requires_grad=True)
    grad_sigma = torch.tensor([[5.0, 2.0], [-1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    eps = gaussian_fourth_moment_rule()
    advantage, _, _ = trainer.hermite_advantage(eps, log_std.exp(), grad_mu, grad_sigma)
    raw_u = (mu + log_std.exp() * eps).detach()
    loss, _ = trainer.delight_score_loss(raw_u, mu, log_std, advantage, objective='pg')
    gradients = torch.autograd.grad(loss, (mu, log_std, grad_mu, grad_sigma), allow_unused=True)
    torch.testing.assert_close(gradients[0], -grad_mu.detach() / len(mu))
    torch.testing.assert_close(gradients[1], -log_std.detach().exp() * grad_sigma.detach() / len(mu))
    assert gradients[2:] == (None, None)
    torch.testing.assert_close(advantage.mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)


def test_hermite_coefficients_and_sampling_context_are_all_detached():
    tensors = [torch.ones(2, 1, 1, requires_grad=True)]
    tensors += [torch.ones(1, 1, requires_grad=True) for _ in range(3)]
    outputs = trainer.hermite_advantage(*tensors)
    assert all(not output.requires_grad for output in outputs)


@pytest.mark.parametrize('order', [1, 2])
def test_constant_value_has_zero_score_update(order):
    mu = torch.tensor([[0.7]], requires_grad=True)
    log_std = torch.tensor([[-0.2]], requires_grad=True)
    eps = torch.tensor([[[1.0]], [[-1.0]]])
    advantage, _, _ = trainer.hermite_advantage(eps, log_std.exp(), torch.zeros_like(mu), torch.zeros_like(mu), order)
    loss, _ = trainer.delight_score_loss((mu + log_std.exp() * eps).detach(), mu, log_std, advantage)
    for gradient in torch.autograd.grad(loss, (mu, log_std)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_order_one_antithetic_dg_mean_update_degenerates_to_pg():
    mu = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.3]], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([[[0.3]], [[1.1]], [[-0.3]], [[-1.1]]], dtype=torch.float64)
    advantage, first, second = trainer.hermite_advantage(eps, log_std.exp(), torch.ones_like(mu), torch.full_like(mu, 3), 1)
    torch.testing.assert_close(advantage, first)
    torch.testing.assert_close(second, torch.zeros_like(second))
    raw_u = (mu + log_std.exp() * eps).detach()
    losses = [trainer.delight_score_loss(raw_u, mu, log_std, advantage, objective=objective)[0]
              for objective in ('pg', 'delight')]
    gradients = [torch.autograd.grad(loss, (mu, log_std), retain_graph=True) for loss in losses]
    for ordinary, gated in zip(*gradients):
        torch.testing.assert_close(ordinary, gated)


def test_order_two_allows_nonzero_dg_mean_correction_without_scale_gating():
    mu = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    log_std = torch.zeros_like(mu, requires_grad=True)
    eps = torch.tensor([[[0.4]], [[1.5]], [[-0.4]], [[-1.5]]], dtype=torch.float64)
    advantage, _, _ = trainer.hermite_advantage(eps, log_std.exp(), torch.ones_like(mu), torch.full_like(mu, 3))
    raw_u = (mu + log_std.exp() * eps).detach()
    losses = [trainer.delight_score_loss(raw_u, mu, log_std, advantage, objective=objective)[0]
              for objective in ('pg', 'delight')]
    gradients = [torch.autograd.grad(loss, (mu, log_std), retain_graph=True) for loss in losses]
    assert not torch.allclose(gradients[0][0], gradients[1][0])
    torch.testing.assert_close(gradients[0][1], gradients[1][1])


def test_total_center_derivatives_match_hermite_jvp_contraction():
    # Three distinct mean routes: latent mu, pre-action mu, and zsa(mu).
    encoder = SimpleNamespace(zsa=lambda z, a: a.square() + z)
    coefficient = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)

    def critic(state, latent, pre, zsa, zs):
        value = latent[:, :1].square() + 3 * latent[:, 1:].square() + 2 * pre + 5 * zsa
        return value * coefficient

    agent = SimpleNamespace(critic=critic, fixed_encoder=encoder)
    mu = torch.tensor([[0.3], [-0.4]], dtype=torch.float64)
    sigma = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    state, zs = torch.ones_like(mu), torch.full_like(mu, 0.1)
    center_fn = lambda m, s: trainer.TD7Agent._center_value_fn(agent, m, s, state, zs)
    with torch.no_grad():
        (grad_mu, grad_sigma), center = torch.func.grad(center_fn, argnums=(0, 1), has_aux=True)(mu, sigma)
    torch.testing.assert_close(grad_mu, (12 * mu + 2) * coefficient.detach().mean())
    torch.testing.assert_close(grad_sigma, 6 * sigma * coefficient.detach().mean())
    eps = torch.tensor([[0.6], [-1.3]], dtype=torch.float64)
    advantage, _, _ = trainer.hermite_advantage(eps, sigma, grad_mu, grad_sigma)
    _, tangent = torch.func.jvp(lambda m, s: center_fn(m, s)[0], (mu, sigma),
                               (sigma * eps, 0.5 * sigma * (eps.square() - 1)))
    torch.testing.assert_close(advantage.sum(), tangent)
    assert not grad_mu.requires_grad and not grad_sigma.requires_grad and not center.requires_grad


def test_native_actor_does_not_query_physical_point_or_cubature():
    source = inspect.getsource(trainer.TD7Agent._hermite_actor_loss_fn)
    tree = ast.parse(textwrap.dedent(source))
    noise_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'randn']
    assert len(noise_calls) == 1
    assert 'self.cubature' not in source
    assert 'flat_action' not in source
    assert 'self._fields' not in source
    assert trainer.Args().actor_advantage == 'hermite'


def test_native_actor_backward_reaches_scores_only(monkeypatch):
    from types import MethodType

    mu = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.4]], dtype=torch.float64, requires_grad=True)
    critic_coefficient = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    encoder_coefficient = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    state = torch.ones_like(mu, requires_grad=True)
    zs = torch.full_like(mu, 0.1, requires_grad=True)
    calls = []

    def critic(s, latent, pre, zsa, z):
        calls.append(latent.shape)
        return (latent[:, :1].square() + latent[:, 1:].square() + pre + zsa) * critic_coefficient

    agent = SimpleNamespace(
        args=trainer.Args(actor_objective='delight'),
        actor=lambda s, z: (mu, log_std), critic=critic,
        fixed_encoder=SimpleNamespace(zsa=lambda z, a: encoder_coefficient * a.square() + z),
    )
    agent._center_value_fn = MethodType(trainer.TD7Agent._center_value_fn, agent)
    eps = torch.tensor([[[0.4]], [[1.5]]], dtype=torch.float64)
    monkeypatch.setattr(torch, 'randn', lambda *shape, device: eps.to(device))
    loss, returned_std, diagnostics = trainer.TD7Agent._hermite_actor_loss_fn(agent, state, zs, torch.tensor(0.0))
    assert returned_std is log_std
    assert diagnostics.shape == (14,)
    assert calls == [torch.Size([1, 2])]  # center only, no 4/12 point-action queries
    gradients = torch.autograd.grad(loss, (mu, log_std, critic_coefficient, encoder_coefficient, state, zs), allow_unused=True)
    assert all(gradient is not None for gradient in gradients[:2])
    assert all(gradient is None for gradient in gradients[2:])
