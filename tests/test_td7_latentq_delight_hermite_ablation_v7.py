"""Independent entropy/target ablation contracts on tensors, without networks."""
import ast
from collections import defaultdict
import inspect
import textwrap
from types import MethodType, SimpleNamespace

import pytest
import torch

from cleanrl.td7.delight import td7_latentq_delight_hermite_v6 as original
from cleanrl.td7.delight import td7_latentq_delight_hermite_ablation_v7 as trainer


def test_disabled_entropy_coefficient_is_exact_zero_and_detached():
    log_alpha = torch.tensor(9.0, requires_grad=True)
    disabled = trainer.effective_entropy_alpha(log_alpha, False)
    assert disabled.item() == 0
    assert not disabled.requires_grad
    torch.testing.assert_close(trainer.effective_entropy_alpha(log_alpha, True), log_alpha.detach().exp())


def tensor_agent(module, entropy_floor=True, objective='delight'):
    mu = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.4]], dtype=torch.float64, requires_grad=True)
    state = torch.ones_like(mu)
    zs = torch.zeros_like(mu)
    args = module.Args(actor_samples=6, actor_objective=objective)
    if module is trainer:
        args.entropy_floor = entropy_floor
    agent = SimpleNamespace(
        args=args, actor=lambda s, z: (mu, log_std),
        critic=lambda s, latent, pre, zsa, z: (latent[:, :1].square() + 2 * latent[:, 1:].square()).expand(-1, 2),
        fixed_encoder=SimpleNamespace(zsa=lambda z, a: a + z),
    )
    agent._center_value_fn = MethodType(module.TD7Agent._center_value_fn, agent)
    agent._hermite_actor_loss_fn = MethodType(module.TD7Agent._hermite_actor_loss_fn, agent)
    return agent, mu, log_std, state, zs


def exact_gaussian_draws(monkeypatch):
    # ±sqrt(3), with four replicated zero nodes: exact second/fourth moments.
    eps = torch.tensor([[[3.0**0.5]], [[0.0]], [[0.0]]], dtype=torch.float64)
    monkeypatch.setattr(torch, 'randn', lambda *shape, device: eps.to(device))


@pytest.mark.parametrize('objective', ['pg', 'delight'])
def test_no_floor_preserves_natural_scale_pg_and_removes_only_entropy(monkeypatch, objective):
    exact_gaussian_draws(monkeypatch)
    gradients = []
    alpha = torch.tensor(2.7, dtype=torch.float64)
    for enabled in (False, True):
        agent, mu, log_std, state, zs = tensor_agent(trainer, enabled, objective)
        # Even a supplied nonzero alpha is ignored by the no-floor actor entry.
        loss, _, _ = trainer.TD7Agent._actor_loss_fn(agent, state, zs, alpha)
        gradients.append(torch.autograd.grad(loss, (mu, log_std)))
    sigma = log_std.detach().exp()
    torch.testing.assert_close(gradients[0][1], -4 * sigma.square())
    torch.testing.assert_close(gradients[1][1], gradients[0][1] - alpha)
    torch.testing.assert_close(gradients[1][0], gradients[0][0])


def test_default_actor_math_and_gradient_match_frozen_v6_exactly(monkeypatch):
    exact_gaussian_draws(monkeypatch)
    values = []
    for module in (original, trainer):
        agent, mu, log_std, state, zs = tensor_agent(module)
        output = module.TD7Agent._actor_loss_fn(agent, state, zs, torch.tensor(1.3, dtype=torch.float64))
        gradients = torch.autograd.grad(output[0], (mu, log_std))
        values.append((output, gradients))
    for before, after in zip(values[0][0] + values[0][1], values[1][0] + values[1][1]):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_entropy_and_target_smoothing_flags_are_independent():
    no_floor = trainer.Args(entropy_floor=False)
    no_smoothing = trainer.Args(target_smoothing_sigma=0.0)
    assert no_floor.target_smoothing_sigma == original.Args().target_smoothing_sigma == 0.2
    assert no_smoothing.entropy_floor
    assert no_floor.target_sigma == no_smoothing.target_sigma == 0.1
    # Target construction, TD/Hermite losses and replay upload are byte-for-byte
    # the same AST as the frozen trainer; the scalar smoothing argument is reused.
    for name in ('_critic_loss_fn', '_encoder_loss_fn', 'flush', '_hermite_actor_loss_fn'):
        before = ast.parse(textwrap.dedent(inspect.getsource(getattr(original.TD7Agent, name))))
        after = ast.parse(textwrap.dedent(inspect.getsource(getattr(trainer.TD7Agent, name))))
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


def test_disabled_alpha_empty_adam_state_restores_without_inserting_entries():
    parameter = torch.tensor(0.0)
    optimizer = SimpleNamespace(
        param_groups=[{'params': [parameter], 'lr': 0.001, 'capturable': True}],
        state=defaultdict(dict),
    )
    saved = {'param_groups': [{'params': [0], 'lr': 0.001, 'capturable': True}], 'state': {}}
    trainer.restore_optimizer_in_place(optimizer, saved)
    assert dict(optimizer.state) == {}
    assert 'alpha_optimizer' in trainer.CHECKPOINT_OPTIMIZERS


def test_enabled_alpha_restore_keeps_captured_state_addresses():
    parameter = torch.tensor(0.0)
    step = torch.tensor(4.0)
    moment = torch.tensor(2.0)
    optimizer = SimpleNamespace(
        param_groups=[{'params': [parameter], 'lr': 0.001}],
        state={parameter: {'step': step, 'exp_avg': moment}},
    )
    saved = {'param_groups': [{'params': [0], 'lr': 0.001}],
             'state': {0: {'step': torch.tensor(9.0), 'exp_avg': torch.tensor(7.0)}}}
    addresses = (step.data_ptr(), moment.data_ptr())
    trainer.restore_optimizer_in_place(optimizer, saved)
    assert addresses == (step.data_ptr(), moment.data_ptr())
    assert step.item() == 9 and moment.item() == 7
